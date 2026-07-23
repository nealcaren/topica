"""Committed-gold parity for topica GDMR vs tomotopy ``GDMRModel`` (issue #271, Wave 1).

topica and tomotopy are independent implementations of the Lee & Song (2020)
g-DMR model (Legendre-polynomial basis over continuous metadata, collapsed Gibbs
sampling). They share no code and no RNG, so agreement is *statistical*: fit both
on the SAME tokenized corpus + metadata and ask whether they recover the same
topic-over-metadata response (the ``tdf`` curve) and the same topic-word content.

The corpus is the synthetic two-cluster fixture from the live parity script
``parity/test_gdmr_tomotopy.py``: 300 short documents whose 1-D metadata in
``[0, 1]`` switches the generating vocabulary (metadata > 0.5 -> "space" words,
else "animal" words). This is a well-identified design where the space topic's
tdf must rise monotonically with the metadata, so both engines agree sharply.

The signature observable is the space topic: its tdf curve over the metadata grid
and its top-word content. The committed gold freezes tomotopy's space-topic tdf
curve, its space-topic word distribution, the vocab, and tomotopy's own
seed-to-seed tdf noise floor, so the bar is benchmarked against tomotopy's own
reproducibility rather than an invented threshold.

Two phases (mirrors parity/stm_gold.py / keyatm_gold.py exactly):

  * ``--regenerate`` (needs tomotopy): fits tomotopy ``GDMRModel`` twice (two
    seeds) to measure its tdf noise floor, freezes one run's tdf curve + space
    topic-word, and writes the committed gold (``parity/gdmr_gold.npz`` + ``.json``).
  * default (no tomotopy): loads the committed gold, fits topica GDMR on the same
    corpus + metadata, aligns its space topic to tomotopy's, and checks the bar.

Run directly::

    python parity/gdmr_gold.py               # offline compare against committed gold
    python parity/gdmr_gold.py --regenerate  # run tomotopy once, write the gold
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness

NAME = "gdmr"

# --------------------------------------------------------------------------- #
# Corpus + config (taken verbatim from parity/test_gdmr_tomotopy.py)
# --------------------------------------------------------------------------- #
_VOCAB_A = ["planet", "star", "moon", "rocket", "orbit"]   # space words
_VOCAB_B = ["cat", "dog", "fish", "bird", "mouse"]         # animal words

NUM_TOPICS = 2
SEED = 0
N = 300
DOC_LEN = 10
DEGREES = [2]   # 1-D metadata, Legendre degree 2
ITERS = 500
SIGMA = 1.0
SIGMA0 = 3.0
# decay > 0 exercises the per-dimension higher-order shrinkage (#426): with sigma
# != sigma0 AND decay > 0 this parity distinguishes the corrected prior from the
# old sigma/sigma0-swapped, geometric-decay one, which a decay=0 config could not.
DECAY = 0.5
BURN_IN = 100

# tdf evaluation grid (verbatim from the live script).
EVAL_XS = np.linspace(0.05, 0.95, 20)

# Pearson-correlation pass margin below tomotopy's own seed-to-seed tdf floor.
MARGIN = 0.15


def _make_corpus(n=N, doc_length=DOC_LEN, seed=SEED):
    """Synthetic two-cluster corpus; metadata > 0.5 draws space words."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, size=n)
    docs = []
    for x in xs:
        vocab = _VOCAB_A if x > 0.5 else _VOCAB_B
        docs.append(rng.choice(vocab, size=doc_length).tolist())
    return docs, xs.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# tomotopy reference
# --------------------------------------------------------------------------- #
def _tomotopy_available() -> bool:
    try:
        import tomotopy  # noqa: F401
        return True
    except Exception:
        return False


def _tomotopy_version() -> str:
    try:
        import tomotopy
        return f"tomotopy {tomotopy.__version__}"
    except Exception:
        return "tomotopy (version unknown)"


def _fit_tomotopy(docs, metadata, seed):
    import tomotopy as tp

    mdl = tp.GDMRModel(
        tw=tp.TermWeight.ONE, k=NUM_TOPICS, degrees=DEGREES,
        sigma=SIGMA, sigma0=SIGMA0, decay=DECAY, seed=seed, min_cf=0, min_df=0,
    )
    for doc, x in zip(docs, metadata[:, 0].tolist()):
        mdl.add_doc(doc, numeric_metadata=[float(x)])
    mdl.burn_in = BURN_IN
    mdl.train(ITERS, show_progress=False)
    return mdl


def _tomotopy_tdf(mdl, xs):
    """tdf curve over the grid; returns (len(xs), K), row-normalized."""
    curves = []
    for x in xs:
        probs = np.array(mdl.tdf([float(x)]))
        probs = probs / probs.sum()
        curves.append(probs)
    return np.stack(curves, axis=0)


def _tomotopy_space_topic(mdl):
    vocab = list(mdl.used_vocabs)
    tw = np.array([mdl.get_topic_word_dist(k) for k in range(NUM_TOPICS)])
    mass = [sum(tw[t, vocab.index(w)] for w in _VOCAB_A if w in vocab)
            for t in range(NUM_TOPICS)]
    return int(np.argmax(mass))


def _tomotopy_space_word_dist(mdl, space_idx):
    """Space-topic word distribution aligned to ``_VOCAB_A + _VOCAB_B`` order."""
    vocab = list(mdl.used_vocabs)
    row = np.array(mdl.get_topic_word_dist(space_idx))
    full = _VOCAB_A + _VOCAB_B
    out = np.zeros(len(full))
    for j, w in enumerate(full):
        if w in vocab:
            out[j] = row[vocab.index(w)]
    return out


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _fit_topica(docs, metadata):
    import topica

    m = topica.GDMR(
        num_topics=NUM_TOPICS, degrees=DEGREES, sigma=SIGMA, sigma0=SIGMA0,
        decay=DECAY, seed=SEED, optimize_interval=25, burn_in=BURN_IN,
    )
    m.fit(docs, metadata, iters=ITERS, num_samples=5, sample_interval=20)
    return m


def _topica_tdf(model, xs):
    pts = np.asarray(xs).reshape(-1, 1)
    return model.tdf(pts, normalize=True)


def _topica_space_topic(model):
    vocab = model.vocabulary
    tw = model.topic_word
    mass = [sum(tw[t, vocab.index(w)] for w in _VOCAB_A if w in vocab)
            for t in range(model.num_topics)]
    return int(np.argmax(mass))


def _topica_space_word_dist(model, space_idx):
    vocab = model.vocabulary
    row = np.asarray(model.topic_word)[space_idx]
    full = _VOCAB_A + _VOCAB_B
    out = np.zeros(len(full))
    for j, w in enumerate(full):
        if w in vocab:
            out[j] = row[vocab.index(w)]
    return out


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not _tomotopy_available():
        print("tomotopy not available; cannot regenerate.")
        sys.exit(1)

    docs, metadata = _make_corpus()

    # Two tomotopy seeds: one frozen as gold, the pair for the noise floor.
    m1 = _fit_tomotopy(docs, metadata, seed=SEED)
    m2 = _fit_tomotopy(docs, metadata, seed=SEED + 100)

    s1 = _tomotopy_space_topic(m1)
    s2 = _tomotopy_space_topic(m2)
    tdf1 = _tomotopy_tdf(m1, EVAL_XS)[:, s1]
    tdf2 = _tomotopy_tdf(m2, EVAL_XS)[:, s2]
    tomotopy_self_r = float(np.corrcoef(tdf1, tdf2)[0, 1])
    word1 = _tomotopy_space_word_dist(m1, s1)

    # topica fit summary captured at regenerate time for the provenance log.
    tm = _fit_topica(docs, metadata)
    t_space = _topica_space_topic(tm)
    t_tdf = _topica_tdf(tm, EVAL_XS)[:, t_space]
    t_word = _topica_space_word_dist(tm, t_space)
    topica_r = float(np.corrcoef(tdf1, t_tdf)[0, 1])
    word_cos = float(
        np.dot(word1, t_word)
        / ((np.linalg.norm(word1) * np.linalg.norm(t_word)) + 1e-12)
    )

    harness.save_gold(
        NAME,
        arrays={
            "space_tdf": tdf1.astype(np.float64),
            "space_word": word1.astype(np.float64),
            "eval_xs": EVAL_XS.astype(np.float64),
            "full_vocab": np.array(_VOCAB_A + _VOCAB_B, dtype=object),
        },
        meta={
            "reference": _tomotopy_version(),
            "model": "GDMR (g-DMR, Lee & Song 2020)",
            "corpus": ("synthetic two-cluster, 300 docs x 10 tokens, 1-D metadata "
                       "in [0,1] switching space/animal vocab (from "
                       "parity/test_gdmr_tomotopy.py)"),
            "num_docs": len(docs),
            "num_topics": NUM_TOPICS,
            "degrees": DEGREES,
            "sigma": SIGMA,
            "sigma0": SIGMA0,
            "iters": ITERS,
            "burn_in": BURN_IN,
            "seeds": {"gold": SEED, "noise_floor": SEED + 100},
            "margin": MARGIN,
            "tomotopy_tdf_self_r": tomotopy_self_r,
            "topica_tdf_r": topica_r,
            "topica_word_cosine": word_cos,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("topica space-topic tdf Pearson r vs tomotopy >= "
                         "tomotopy_tdf_self_r - margin, AND space-word cosine >= 0.9"),
            "kind": "cross-implementation (tomotopy GDMRModel)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  tomotopy tdf self-r : {tomotopy_self_r:.4f}")
    print(f"  topica  tdf r       : {topica_r:.4f}")
    print(f"  topica  word cosine : {word_cos:.4f}")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    gold_tdf = arrays["space_tdf"]
    gold_word = arrays["space_word"]
    xs = arrays["eval_xs"]
    margin = float(meta.get("margin", MARGIN))
    self_r = float(meta.get("tomotopy_tdf_self_r", 0.0))

    docs, metadata = _make_corpus()
    tm = _fit_topica(docs, metadata)
    t_space = _topica_space_topic(tm)
    t_tdf = _topica_tdf(tm, xs)[:, t_space]
    t_word = _topica_space_word_dist(tm, t_space)

    tdf_r = float(np.corrcoef(gold_tdf, t_tdf)[0, 1])
    word_cos = float(
        np.dot(gold_word, t_word)
        / ((np.linalg.norm(gold_word) * np.linalg.norm(t_word)) + 1e-12)
    )
    bar = self_r - margin
    result = {
        "tdf_r": tdf_r,
        "word_cosine": word_cos,
        "tomotopy_tdf_self_r": self_r,
        "bar": bar,
        "margin_over_bar": tdf_r - bar,
        "passes": bool(tdf_r >= bar and word_cos >= 0.9),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica tdf r vs tomotopy : {tdf_r:.4f} "
              f"(tomotopy self {self_r:.4f}, bar {bar:.4f})")
        print(f"  space-word cosine        : {word_cos:.4f} (bar 0.9)")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(tdf margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
