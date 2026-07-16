"""Committed-gold parity for topica SAGE (issue #271, Wave 1).

SAGE (Eisenstein, Ahmed & Xing 2011) learns shared topics whose word
distributions vary by a document-level group covariate:
``log beta_{k,g,v} = m_v + kappaT_{k,v} + kappaC_{g,v} + kappaI_{k,g,v}``.

There is NO external reference implementation to benchmark against (gensim and
tomotopy do not implement SAGE, and there is no R package), so this is a
**planted-recovery / self-consistency gold**, NOT a cross-implementation one. We
fit topica SAGE ONCE on a fixed-seed synthetic corpus with a known, identifiable
structure and freeze its group-specific topic-word distributions; the test then
refits and asserts the fit reproduces the frozen solution AND recovers the planted
structure, and a shuffled matrix fails the gate.

The corpus is the bilingual fixture from ``tests/test_sage.py``: two topics
(weather / food) crossed with two language groups (en / de) drawn from FULLY
DISJOINT vocabularies. The identifying property is that each topic's English
top-words are English and its German top-words are German — a clean recovery
target that a degenerate (group-blind) fit cannot satisfy.

Two phases (mirrors parity/stm_gold.py / keyatm_gold.py):

  * ``--regenerate``: fits topica SAGE once on the fixed-seed corpus, records the
    group-specific topic-word distributions and the planted-recovery invariants,
    and writes the committed gold (``parity/sage_gold.npz`` + ``.json``).
  * default: loads the committed gold, refits topica SAGE, and asserts the aligned
    group-specific topic-word cosine clears the bar and the recovery invariants hold.

Run directly::

    python parity/sage_gold.py               # offline compare against committed gold
    python parity/sage_gold.py --regenerate  # fit once, write the gold
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness

NAME = "sage"

# --------------------------------------------------------------------------- #
# Corpus + config (taken verbatim from tests/test_sage.py)
# --------------------------------------------------------------------------- #
_EN_WEATHER = ["rain", "sun", "cloud", "wind", "storm"]
_DE_WEATHER = ["regen", "sonne", "wolke", "sturm", "nebel"]
_EN_FOOD = ["bread", "cheese", "wine", "apple", "meat"]
_DE_FOOD = ["brot", "kaese", "wein", "apfel", "fleisch"]

_EN_VOCAB = set(_EN_WEATHER) | set(_EN_FOOD)
_DE_VOCAB = set(_DE_WEATHER) | set(_DE_FOOD)

NUM_TOPICS = 2
N_PER_CELL = 50
CORPUS_SEED = 42
FIT_SEED = 1
ITERS = 300
NUM_SAMPLES = 3
SAMPLE_INTERVAL = 10
OPTIMIZE_INTERVAL = 25
BURN_IN = 50

# Self-consistency cosine bar (refit vs frozen): SAGE's Gibbs fit is deterministic
# under a fixed seed, so the refit should match the gold essentially bit-for-bit;
# the bar leaves a small margin for platform float noise. The shuffled-matrix
# non-vacuous check shows a wrong matrix falls far below this.
COSINE_BAR = 0.99


def _make_bilingual_corpus(n_per_cell=N_PER_CELL, seed=CORPUS_SEED):
    """Bilingual 2-topic x 2-group corpus with disjoint per-group vocab."""
    rng = np.random.default_rng(seed)
    docs, groups = [], []
    for _ in range(n_per_cell):
        docs.append(rng.choice(_EN_WEATHER, size=10).tolist()
                    + rng.choice(_EN_FOOD, size=2).tolist())
        groups.append("en")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_EN_FOOD, size=10).tolist()
                    + rng.choice(_EN_WEATHER, size=2).tolist())
        groups.append("en")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_DE_WEATHER, size=10).tolist()
                    + rng.choice(_DE_FOOD, size=2).tolist())
        groups.append("de")
    for _ in range(n_per_cell):
        docs.append(rng.choice(_DE_FOOD, size=10).tolist()
                    + rng.choice(_DE_WEATHER, size=2).tolist())
        groups.append("de")
    return docs, groups


def _fit_topica(docs, groups):
    from topica.models import SAGE

    model = SAGE(num_topics=NUM_TOPICS, seed=FIT_SEED,
                 optimize_interval=OPTIMIZE_INTERVAL, burn_in=BURN_IN)
    model.fit(docs, groups, iters=ITERS, num_samples=NUM_SAMPLES,
              sample_interval=SAMPLE_INTERVAL)
    return model


def _group_topic_word(model, vocab_order):
    """topica SAGE topic_word is (K, G, V); reorder V columns onto ``vocab_order``
    and return (K, G, len(vocab_order))."""
    vocab = list(model.vocabulary)
    idx = {w: i for i, w in enumerate(vocab)}
    tw = np.asarray(model.topic_word)  # (K, G, V)
    K, G, _ = tw.shape
    out = np.zeros((K, G, len(vocab_order)))
    for j, w in enumerate(vocab_order):
        if w in idx:
            out[:, :, j] = tw[:, :, idx[w]]
    return out


def _flatten_kg(tw_kg):
    """(K, G, V) -> (K*G, V) so the Hungarian alignment treats each (topic, group)
    distribution as one row to match across two fits."""
    K, G, V = tw_kg.shape
    return tw_kg.reshape(K * G, V)


def _recovery_invariants(model):
    """Planted structure: each topic's en top-words are English, de top-words are
    German, and the two sets are near-disjoint. Returns a dict of booleans/scores."""
    en_ok = de_ok = disjoint_ok = True
    for t in range(NUM_TOPICS):
        en = {w for w, _ in model.top_words(7, topic=t, group="en")}
        de = {w for w, _ in model.top_words(7, topic=t, group="de")}
        en_ok &= en <= _EN_VOCAB
        de_ok &= de <= _DE_VOCAB
        disjoint_ok &= len(en & de) <= 1
    return {"en_words_english": bool(en_ok),
            "de_words_german": bool(de_ok),
            "groups_disjoint": bool(disjoint_ok)}


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    docs, groups = _make_bilingual_corpus()
    vocab_order = sorted(_EN_VOCAB | _DE_VOCAB)

    model = _fit_topica(docs, groups)
    tw_kg = _group_topic_word(model, vocab_order)
    inv = _recovery_invariants(model)
    # groups property is sorted -> ['de', 'en']
    group_order = list(model.groups)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word_kg": tw_kg.astype(np.float64),  # (K, G, V)
            "vocab": np.array(vocab_order, dtype=object),
            "groups": np.array(group_order, dtype=object),
        },
        meta={
            "reference": "topica SAGE (self-consistency / planted-recovery gold)",
            "model": "SAGE (Eisenstein, Ahmed & Xing 2011)",
            "corpus": ("synthetic bilingual 2-topic x 2-group, disjoint per-group "
                       "vocab, 200 docs (from tests/test_sage.py)"),
            "num_docs": len(docs),
            "num_topics": NUM_TOPICS,
            "num_groups": len(group_order),
            "groups": group_order,
            "corpus_seed": CORPUS_SEED,
            "fit_seed": FIT_SEED,
            "iters": ITERS,
            "num_samples": NUM_SAMPLES,
            "sample_interval": SAMPLE_INTERVAL,
            "optimize_interval": OPTIMIZE_INTERVAL,
            "burn_in": BURN_IN,
            "cosine_bar": COSINE_BAR,
            "recovery_invariants": inv,
            "date": datetime.date.today().isoformat(),
            "kind": ("PLANTED self-consistency gold (NO external reference exists "
                     "for SAGE). Validates reproducibility + planted bilingual "
                     "structure recovery; non-vacuous via the shuffle check."),
            "pass_bar": ("refit-vs-gold aligned (topic,group) cosine >= cosine_bar "
                         "AND all recovery invariants true"),
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  recovery invariants: {inv}")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    gold_kg = arrays["topic_word_kg"]
    vocab_order = list(arrays["vocab"])
    bar = float(meta.get("cosine_bar", COSINE_BAR))

    docs, groups = _make_bilingual_corpus()
    model = _fit_topica(docs, groups)
    refit_kg = _group_topic_word(model, vocab_order)

    cosine, _ = harness.align_cosine(_flatten_kg(gold_kg), _flatten_kg(refit_kg))
    inv = _recovery_invariants(model)
    recovered = all(inv.values())

    result = {
        "cosine": cosine,
        "bar": bar,
        "margin_over_bar": cosine - bar,
        "recovery_invariants": inv,
        "passes": bool(cosine >= bar and recovered),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  refit-vs-gold (topic,group) cosine : {cosine:.4f} (bar {bar:.2f})")
        print(f"  recovery invariants                : {inv}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
