"""Committed `ssnmf` gold for topica GuidedNMF (#683).

GuidedNMF (Vendrow, Haddock, Rebrova & Needell, ICASSP 2021) is seed-word-guided
semi-supervised NMF: it factors the nonneg document-term matrix ``X ~ A S`` while a
supervision term ``lam * ||Y - B S||_F^2`` steers some learned topics toward
user-supplied seed-word groups (Y is the seed matrix, one row per guided group).

The reference is the `ssnmf` package (MIT), pinned ``ssnmf==0.0.2`` by the GuidedNMF
repo, run in its supervised Frobenius mode (`snmfmult`, model 2:
``||X-AS||_F^2 + lam ||Y-BS||_F^2``). The GuidedNMF repo itself carries no license,
so only the *construction* (seed words -> Y) is reproduced, not its code.

We freeze, from a fixed-seed planted corpus with a known 4-topic / 3-seed-group
structure:

  1. ssnmf's topic-word S and doc-topic A under its DEFAULT random init at two
     corpus/init seeds -> a topic-word cosine noise floor (how well ssnmf agrees
     with itself across inits), plus the planted-topic recovery bar; and
  2. ssnmf's S, A, B under an EXPLICIT fixed init (A0,S0,B0) -> the exact
     update-math target. Feeding the same init to topica isolates the
     multiplicative-update arithmetic from the RNG and should agree to ~fp noise.

Offline `run()` refits topica GuidedNMF on the same corpus and asserts:

  * with the same explicit init, topica's S matches the ssnmf S to a tight
    tolerance (the update math is faithful), and
  * under topica's own default init, topica's seeded topic-word phi clears the
    ssnmf self-consistency floor and recovers the planted seed structure.

Runs in CI WITHOUT ssnmf: the reference fit is frozen in the committed
``parity/guidednmf_gold.npz`` + ``.json``.

Two phases::

    python parity/guidednmf_gold.py --regenerate   # needs ssnmf==0.0.2
    python parity/guidednmf_gold.py                # offline compare against the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "guidednmf"

# Planted structure: 4 topics, each a block of BLOCK words; 3 seed groups guide
# the first three topics. The 4th topic is left unguided.
N_TOPICS = 4
BLOCK = 8
N_BLOCKS = 4
VOCAB = [f"t{t}_w{j}" for t in range(N_BLOCKS) for j in range(BLOCK)]
V = len(VOCAB)

# Seed groups: name -> seed words (a subset of each block). Guides topics 0,1,2.
SEEDS: dict[str, list[str]] = {
    "alpha": [f"t0_w{j}" for j in range(3)],
    "beta": [f"t1_w{j}" for j in range(3)],
    "gamma": [f"t2_w{j}" for j in range(3)],
}
N_SEEDED = len(SEEDS)

N_DOCS = 80  # 20 per topic
DOC_LEN = 60
CORPUS_SEED = 271
LAM = 20.0
ITERS = 50
SEED_WEIGHT = 1.0
EPS = 1e-10
MARGIN = 0.15


def _planted_corpus(seed: int):
    """Return (token_docs, counts DxV) for a planted 4-topic corpus. Each doc is
    dominated by one topic block with light cross-topic noise."""
    rng = np.random.RandomState(seed)
    docs, counts = [], np.zeros((N_DOCS, V), dtype=float)
    per = N_DOCS // N_BLOCKS
    for d in range(N_DOCS):
        t = d // per
        p = np.full(V, 0.02 / V)  # background
        p[t * BLOCK:(t + 1) * BLOCK] += 1.0
        p = p / p.sum()
        c = rng.multinomial(DOC_LEN, p)
        counts[d] = c
        toks = []
        for j, n in enumerate(c):
            toks.extend([VOCAB[j]] * int(n))
        docs.append(toks)
    return docs, counts


def _seed_matrix() -> np.ndarray:
    """Y: (N_SEEDED x V) seed matrix, row g = SEED_WEIGHT at each seed word."""
    vidx = {w: i for i, w in enumerate(VOCAB)}
    Y = np.zeros((N_SEEDED, V))
    for g, (_name, words) in enumerate(SEEDS.items()):
        for w in words:
            Y[g, vidx[w]] = SEED_WEIGHT
    return Y


def _fit_ssnmf(counts, Y, *, init_seed, lam=LAM, iters=ITERS, explicit=None):
    """Run ssnmf 0.0.2 supervised Frobenius (snmfmult). counts is DxV (docs x
    words) = ssnmf's samples x features; S is (k x V) topic-word, A is (D x k)
    doc-topic, B is (N_SEEDED x k). If `explicit` is (A0,S0,B0), use it as init."""
    from ssnmf import SSNMF

    if explicit is not None:
        A0, S0, B0 = explicit
        model = SSNMF(counts, N_TOPICS, Y=Y, lam=lam, A=A0.copy(), S=S0.copy(), B=B0.copy())
    else:
        rs = np.random.RandomState(init_seed)
        A0 = rs.rand(counts.shape[0], N_TOPICS)
        S0 = rs.rand(N_TOPICS, counts.shape[1])
        B0 = rs.rand(N_SEEDED, N_TOPICS)
        model = SSNMF(counts, N_TOPICS, Y=Y, lam=lam, A=A0.copy(), S=S0.copy(), B=B0.copy())
    model.snmfmult(numiters=iters, saveerrs=True)
    return np.asarray(model.A), np.asarray(model.S), np.asarray(model.B), (A0, S0, B0)


def _explicit_init(seed: int):
    rs = np.random.RandomState(seed)
    A0 = rs.rand(N_DOCS, N_TOPICS)
    S0 = rs.rand(N_TOPICS, V)
    B0 = rs.rand(N_SEEDED, N_TOPICS)
    return A0, S0, B0


def _ssnmf_version() -> str:
    try:
        import ssnmf

        return f"ssnmf {getattr(ssnmf, '__version__', '0.0.2')}"
    except Exception:
        return "ssnmf (version unknown)"


def _align_to_vocab(mat_kv, topica_vocab):
    """Reorder a topica (K x |topica_vocab|) matrix to the gold VOCAB order."""
    tv = {w: i for i, w in enumerate(topica_vocab)}
    out = np.zeros((mat_kv.shape[0], V))
    for j, w in enumerate(VOCAB):
        if w in tv:
            out[:, j] = mat_kv[:, tv[w]]
    return out


def _fit_topica_default(docs):
    """Fit topica GuidedNMF with random init and COUNT weighting (apples-to-apples
    with the count-weighted ssnmf default gold; topica's own product default is
    tfidf, exercised separately in the unit tests). Returns (topic_word aligned to
    VOCAB, seed_topic_indices, seed_group_names)."""
    import topica

    corpus = topica.Corpus.from_documents(docs, vocabulary=VOCAB)
    m = topica.GuidedNMF(N_TOPICS, SEEDS, guidance=LAM, weighting="count",
                         init="random", seed=13)
    m.fit(corpus, iters=ITERS)
    tw = _align_to_vocab(np.asarray(m.topic_word), list(m.vocabulary))
    return tw, list(getattr(m, "seed_topic_indices", [])), list(SEEDS.keys())


def _fit_topica_explicit(docs, init, iters):
    """Fit topica GuidedNMF with the SAME explicit init as ssnmf (raw factors).
    Returns (S raw aligned to VOCAB, A raw DxK, B raw GxK). Requires the
    explicit-init ("none") path in the binding. Uses count weighting to match the
    gold's X exactly and convergence_tol=0.0 for a fixed iteration budget."""
    import topica

    A0, S0, B0 = init
    corpus = topica.Corpus.from_documents(docs, vocabulary=VOCAB)
    m = topica.GuidedNMF(
        N_TOPICS, SEEDS, guidance=LAM, weighting="count",
        init="none", init_a=A0, init_s=S0, init_b=B0,
        seed_weight=SEED_WEIGHT, convergence_tol=0.0,
    )
    m.fit(corpus, iters=iters)
    # Raw (un-normalized) factors: the faithful comparison target.
    S_raw = _align_to_vocab(np.asarray(m.factor_s), list(m.vocabulary))
    return S_raw, np.asarray(m.factor_a), np.asarray(m.factor_b)


def regenerate() -> None:
    try:
        import ssnmf  # noqa: F401
    except Exception:
        raise SystemExit("regenerate needs ssnmf==0.0.2 (pip install ssnmf==0.0.2)")

    Y = _seed_matrix()
    docs, counts = _planted_corpus(CORPUS_SEED)

    # (1) default-init self-consistency floor: two independent random inits.
    _A1, S1, _B1, _ = _fit_ssnmf(counts, Y, init_seed=1)
    _A2, S2, _B2, _ = _fit_ssnmf(counts, Y, init_seed=2)
    self_cos, _ = harness.align_cosine(S1, S2)  # ssnmf-vs-ssnmf topic-word cosine

    # (2) explicit-init exact-math targets: one update (raw factors, ~1e-9 target)
    # and the full 50-iter run (aligned cosine target). Same frozen init for both.
    init = _explicit_init(7)
    A1, S1_e, B1, _ = _fit_ssnmf(counts, Y, init_seed=0, iters=1, explicit=init)
    A_e, S_e, B_e, _ = _fit_ssnmf(counts, Y, init_seed=0, iters=ITERS, explicit=init)

    harness.save_gold(
        NAME,
        arrays={
            "S_default": S1,        # topic-word under default init seed=1
            "A_default": _A1,       # doc-topic under default init seed=1
            "A_iter1": A1,          # raw factors after EXACTLY one update (explicit init)
            "S_iter1": S1_e,
            "B_iter1": B1,
            "S_explicit": S_e,      # topic-word after ITERS updates (explicit init)
            "A_explicit": A_e,
            "B_explicit": B_e,
            "init_A": init[0],
            "init_S": init[1],
            "init_B": init[2],
            "counts": counts,
            "Y": Y,
            "vocab": np.array(VOCAB, dtype=object),
        },
        meta={
            "reference": _ssnmf_version(),
            "model": "GuidedNMF (jvendrow/GuidedNMF on ssnmf==0.0.2, model 2 Frobenius)",
            "corpus": f"planted {N_DOCS}-doc / {N_BLOCKS}-topic corpus (seed {CORPUS_SEED})",
            "seeds": SEEDS,
            "num_topics": N_TOPICS,
            "num_seeded": N_SEEDED,
            "lam": LAM,
            "iters": ITERS,
            "seed_weight": SEED_WEIGHT,
            "eps": EPS,
            "vocab_size": V,
            "num_docs": N_DOCS,
            "margin": MARGIN,
            "ssnmf_self_topic_cosine": self_cos,
            "date": datetime.date.today().isoformat(),
            "pass_bar": "explicit-init topica S ~= ssnmf S (max|Δ| small) AND "
            "default-init topica topic-word cosine >= ssnmf_self - margin",
        },
    )
    print(f"regenerated {NAME} gold:")
    print(f"  ssnmf self topic-word cosine (init1 vs init2): {self_cos:.4f}")
    print(f"  explicit-init S range: [{S_e.min():.4g}, {S_e.max():.4g}]")


def run(verbose: bool = True) -> dict:
    """Four checks (Gate-A/B design):
      (1) 1-iteration explicit-init RAW factors ~= ssnmf (max|Δ| < 1e-8): the
          update math (orientation, order, eps, λ placement) is faithful.
      (2) 50-iteration explicit-init RAW factors ~= ssnmf (max|Δ| < 1e-6) and
          aligned topic-word cosine >= 0.999: the math stays faithful over the full
          run (raw agreement holds because the same init fixes the topic order; the
          tiny drift is rayon-vs-BLAS float order).
      (3) default-init (random, COUNT weighting) topic-word clears the ssnmf
          self-consistency floor: aligned cosine vs the count-weighted ssnmf default
          gold >= self_cos - margin.
      (4) default-init planted recovery: each seed group steers the topic carrying
          its planted block."""
    arrays, meta = harness.load_gold(NAME)
    self_cos = float(meta["ssnmf_self_topic_cosine"])
    margin = float(meta["margin"])
    bar = self_cos - margin
    docs, _counts = _planted_corpus(CORPUS_SEED)
    init = (arrays["init_A"], arrays["init_S"], arrays["init_B"])

    # (1) one-update raw-factor check.
    S1, A1, B1 = _fit_topica_explicit(docs, init, iters=1)
    iter1_max = max(
        float(np.abs(S1 - arrays["S_iter1"]).max()),
        float(np.abs(A1 - arrays["A_iter1"]).max()),
        float(np.abs(B1 - arrays["B_iter1"]).max()),
    )
    iter1_ok = iter1_max < 1e-8

    # (2) full-run raw agreement + aligned cosine. Same explicit init fixes the
    # topic order, so raw factors are directly comparable (no permutation).
    S50, A50, B50 = _fit_topica_explicit(docs, init, iters=ITERS)
    iter50_max = max(
        float(np.abs(S50 - arrays["S_explicit"]).max()),
        float(np.abs(A50 - arrays["A_explicit"]).max()),
        float(np.abs(B50 - arrays["B_explicit"]).max()),
    )
    cos50, _ = harness.align_cosine(arrays["S_explicit"], S50)
    iter50_ok = iter50_max < 1e-6 and cos50 >= 0.999

    # (3) default-init floor: count-vs-count aligned cosine >= ssnmf self floor.
    tw, seed_idx, names = _fit_topica_default(docs)
    floor_cos, _ = harness.align_cosine(arrays["S_default"], tw)
    floor_ok = floor_cos >= bar

    # (4) default-init planted recovery: each seed group -> topic on its block.
    recovery_ok = True
    steering = {}
    for g, name in enumerate(names):
        block = int(SEEDS[name][0].split("_")[0][1:])  # "t0_w0" -> 0
        k = seed_idx[g] if g < len(seed_idx) else int(np.argmax([tw[kk].sum() for kk in range(N_TOPICS)]))
        top = np.argsort(tw[k])[::-1][:BLOCK]
        hit = sum(1 for j in top if VOCAB[j].startswith(f"t{block}_"))
        steering[name] = (k, hit)
        recovery_ok = recovery_ok and hit >= BLOCK // 2

    result = {
        "iter1_max_abs_diff": iter1_max,
        "iter50_max_abs_diff": iter50_max,
        "explicit_cosine_50": cos50,
        "default_floor_cosine": floor_cos,
        "ssnmf_self_topic_cosine": self_cos,
        "bar": bar,
        "steering": steering,
        "passes": bool(iter1_ok and iter50_ok and floor_ok and recovery_ok),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"(1) 1-iter raw factors: max|Δ| = {iter1_max:.2e} "
              f"-> {'FAITHFUL' if iter1_ok else 'MISMATCH'}")
        print(f"(2) 50-iter raw factors: max|Δ| = {iter50_max:.2e}, aligned cosine = "
              f"{cos50:.6f} -> {'FAITHFUL' if iter50_ok else 'DRIFT'}")
        print(f"(3) default-init floor: cosine = {floor_cos:.4f} "
              f"(bar {bar:.4f}, ssnmf self {self_cos:.4f}) -> {'OK' if floor_ok else 'LOW'}")
        print(f"(4) default-init planted recovery: {steering} "
              f"-> {'OK' if recovery_ok else 'FAIL'}")
        print(f"verdict: {'PASS' if result['passes'] else 'FAIL'}")
    return result


def _row_norm(m: np.ndarray) -> np.ndarray:
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return m / s


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
