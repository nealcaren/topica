"""Committed-gold parity for topica DTM vs gensim ``LdaSeqModel`` (issue #271, Wave 1).

topica and gensim are independent implementations of the Blei & Lafferty (2006)
Dynamic Topic Model (topic-word chains evolving as a Gaussian random walk over
time slices, logistic-normal variational inference). They share no code and no
RNG, so agreement is *statistical*: fit both on the SAME tokenized corpus + time
slices and ask whether their per-time-slice topic-word distributions match.

The corpus is a synthetic SMOOTH-drift design: across four time slices, one
topic's word mass slides gradually along ``w0..w9`` (a moving Gaussian center)
while a second topic stays anchored on ``w10..w13``. This is the Gaussian-random-
walk regime DTM is built for, so the two engines agree sharply (per-slice,
per-topic cosine ~0.999). An ABRUPT planted drift (e.g. tests/test_dtm.py's
{0,1,2}->{2,3,4}->{4,5,6}) is NOT well-identified across implementations: the two
variational chains move the drifting topic at different rates and the drift topic
disagrees badly even though the stable topic agrees. We therefore use the smooth
design, where absolute agreement is high and the bar is meaningful.

The committed gold freezes gensim's per-slice topic-word distributions (aligned to
a shared vocab), the time slices, and the exact corpus, so the offline test refits
topica on the identical corpus and aligns to gensim.

Two phases (mirrors parity/stm_gold.py):

  * ``--regenerate`` (needs gensim): fits gensim ``LdaSeqModel`` once, freezes its
    per-slice topic-word array + the corpus, and writes the committed gold
    (``parity/dtm_gold.npz`` + ``.json``).
  * default (no gensim): loads the committed gold, fits topica DTM on the same
    corpus, aligns per-slice to gensim's topics, and checks the bar.

Run directly::

    python parity/dtm_gold.py               # offline compare against committed gold
    python parity/dtm_gold.py --regenerate  # run gensim once, write the gold
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness

NAME = "dtm"

NUM_TOPICS = 2
NUM_TIMES = 4
N_PER_CELL = 150
DOC_LEN = 12
CORPUS_SEED = 0
CHAIN_VARIANCE = 0.1
TOPICA_ITERS = 40
GENSIM_PASSES = 15
GENSIM_SEED = 1
# Stable topic anchor and drift band (word ids in the vocab w0..w19).
_DRIFT_BAND = list(range(10))      # w0..w9
_STABLE_WORDS = [10, 11, 12, 13]   # w10..w13
VOCAB = [f"w{i}" for i in range(20)]

# Per-slice, per-topic cosine pass bar. On the smooth-drift design both engines
# agree at ~0.999, so a 0.90 bar is cleared by a wide margin while still failing
# a shuffled / wrong matrix (shown by the non-vacuous test).
COSINE_BAR = 0.90


def _make_corpus(seed=CORPUS_SEED):
    """Smooth-drift corpus: drift topic mass slides along w0..w9 across slices,
    stable topic anchored on w10..w13. Returns (docs, times)."""
    rng = np.random.default_rng(seed)
    docs, times = [], []
    for s in range(NUM_TIMES):
        center = 1.5 + 6.0 * s / (NUM_TIMES - 1)  # 1.5 -> 7.5
        for _ in range(N_PER_CELL):
            idx = np.clip(np.round(rng.normal(center, 1.0, size=DOC_LEN)).astype(int),
                          0, len(_DRIFT_BAND) - 1)
            docs.append([VOCAB[i] for i in idx])
            times.append(s)
            docs.append([VOCAB[i] for i in rng.choice(_STABLE_WORDS, size=DOC_LEN)])
            times.append(s)
    return docs, times


# --------------------------------------------------------------------------- #
# gensim reference
# --------------------------------------------------------------------------- #
def _gensim_available() -> bool:
    try:
        import gensim.corpora  # noqa: F401
        import gensim.models   # noqa: F401
        return True
    except Exception:
        return False


def _gensim_version() -> str:
    try:
        import gensim
        return f"gensim {gensim.__version__}"
    except Exception:
        return "gensim (version unknown)"


def _fit_gensim(docs, times):
    """Fit gensim LdaSeqModel; return a (K, T, V) topic-word array aligned to VOCAB."""
    from gensim.corpora import Dictionary
    from gensim.models import LdaSeqModel

    order = sorted(range(len(times)), key=lambda i: times[i])
    docs_sorted = [docs[i] for i in order]
    time_slice = [sum(1 for t in times if t == s) for s in range(NUM_TIMES)]
    dct = Dictionary(docs_sorted)
    corpus = [dct.doc2bow(d) for d in docs_sorted]
    m = LdaSeqModel(corpus=corpus, id2word=dct, time_slice=time_slice,
                    num_topics=NUM_TOPICS, chain_variance=CHAIN_VARIANCE,
                    passes=GENSIM_PASSES, random_state=GENSIM_SEED)

    out = np.zeros((NUM_TOPICS, NUM_TIMES, len(VOCAB)))
    for k in range(NUM_TOPICS):
        for t in range(NUM_TIMES):
            p = np.exp(m.topic_chains[k].e_log_prob[:, t])
            p = p / p.sum()
            for wid, w in enumerate(VOCAB):
                if w in dct.token2id:
                    out[k, t, wid] = p[dct.token2id[w]]
    return out


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _fit_topica(docs, times):
    """Fit topica DTM; return a (K, T, V) topic-word array aligned to VOCAB."""
    import topica

    m = topica.DTM(num_topics=NUM_TOPICS, chain_variance=CHAIN_VARIANCE,
                   seed=GENSIM_SEED)
    m.fit(docs, times, iters=TOPICA_ITERS)
    tv = list(m.vocabulary)
    idx = {w: i for i, w in enumerate(tv)}
    out = np.zeros((NUM_TOPICS, m.num_times, len(VOCAB)))
    for t in range(m.num_times):
        row = np.asarray(m.topic_word(t))  # (K, V_topica)
        for wid, w in enumerate(VOCAB):
            if w in idx:
                out[:, t, wid] = row[:, idx[w]]
    return out


def _aligned_per_slice_cosine(gold_ktv, refit_ktv):
    """Align topics by their first-slice rows, then mean per-slice/per-topic cosine.
    Returns (mean_cosine, min_cosine, perm)."""
    g0 = gold_ktv[:, 0, :]
    r0 = refit_ktv[:, 0, :]
    _, perm = harness.align_cosine(g0, r0)
    refit = refit_ktv[perm]
    cos = []
    for k in range(gold_ktv.shape[0]):
        for t in range(gold_ktv.shape[1]):
            a, b = gold_ktv[k, t], refit[k, t]
            cos.append(float(np.dot(a, b)
                             / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(cos)), float(np.min(cos)), perm


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not _gensim_available():
        print("gensim not available; cannot regenerate.")
        sys.exit(1)

    docs, times = _make_corpus()
    gold_ktv = _fit_gensim(docs, times)

    # topica fit summary captured at regenerate time for the provenance log.
    refit_ktv = _fit_topica(docs, times)
    mean_cos, min_cos, _ = _aligned_per_slice_cosine(gold_ktv, refit_ktv)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word_ktv": gold_ktv.astype(np.float32),  # (K, T, V)
            "vocab": np.array(VOCAB, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "times": np.array(times, dtype=np.int32),
        },
        meta={
            "reference": _gensim_version(),
            "model": "DTM (Blei & Lafferty 2006)",
            "corpus": ("synthetic smooth-drift, 4 time slices x 300 docs, drift "
                       "topic mass sliding along w0..w9, stable topic on w10..w13"),
            "num_docs": len(docs),
            "num_topics": NUM_TOPICS,
            "num_times": NUM_TIMES,
            "vocab_size": len(VOCAB),
            "chain_variance": CHAIN_VARIANCE,
            "topica_iters": TOPICA_ITERS,
            "gensim_passes": GENSIM_PASSES,
            "seeds": {"corpus": CORPUS_SEED, "gensim": GENSIM_SEED,
                      "topica": GENSIM_SEED},
            "cosine_bar": COSINE_BAR,
            "topica_mean_cosine": mean_cos,
            "topica_min_cosine": min_cos,
            "date": datetime.date.today().isoformat(),
            "kind": "cross-implementation (gensim LdaSeqModel)",
            "pass_bar": ("topica per-slice/per-topic topic-word cosine vs gensim, "
                         "minimum across all (topic, slice) >= cosine_bar"),
            "note": ("Uses a SMOOTH Gaussian-random-walk drift, the regime DTM "
                     "models. An abrupt planted drift is not well-identified "
                     "across implementations (the drifting chain disagrees)."),
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  topica vs gensim mean cosine {mean_cos:.4f}  min {min_cos:.4f}")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    gold_ktv = arrays["topic_word_ktv"].astype(np.float64)
    bar = float(meta.get("cosine_bar", COSINE_BAR))

    docs = harness.lines_to_docs(str(arrays["corpus"]))
    times = arrays["times"].tolist()
    refit_ktv = _fit_topica(docs, times)

    mean_cos, min_cos, _ = _aligned_per_slice_cosine(gold_ktv, refit_ktv)
    result = {
        "mean_cosine": mean_cos,
        "min_cosine": min_cos,
        "bar": bar,
        "margin_over_bar": min_cos - bar,
        "passes": bool(min_cos >= bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica vs gensim mean cosine : {mean_cos:.4f}")
        print(f"  minimum (topic, slice) cosine: {min_cos:.4f} (bar {bar:.2f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
