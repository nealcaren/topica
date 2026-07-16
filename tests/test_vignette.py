"""End-to-end regression test: the STM vignette workflow on the gadarian data.

Guards the full pipeline — preprocess -> STM(prevalence) -> estimate_effect —
and the substantive finding it reproduces: the experimental `treatment`
significantly shifts topic prevalence (Gadarian & Albertson 2014). Skips if the
vendored dataset is absent.
"""

import csv
import os
from collections import Counter

import numpy as np
import pytest

from topica.models import STM
from topica import tokenize, stm

GADARIAN = os.path.join(os.path.dirname(__file__), "..", "examples", "gadarian.csv")
STOPLIST = os.path.join(os.path.dirname(__file__), "..", "examples", "english-stoplist.txt")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(GADARIAN) and os.path.exists(STOPLIST)),
    reason="gadarian dataset / stoplist not present",
)


def _load_and_prep():
    with open(GADARIAN, newline="") as f:
        rows = list(csv.DictReader(f))
    text = [r["open.ended.response"] for r in rows]
    treatment = np.array([float(r["treatment"]) for r in rows])
    pid = np.array([float(r["pid_rep"]) for r in rows])
    stopwords = open(STOPLIST).read().split()

    toks = [tokenize(t, stopwords=stopwords, min_length=3) for t in text]
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= 3}
    toks = [[w for w in d if w in vocab] for d in toks]
    keep = np.array([len(d) > 0 for d in toks])
    docs = [d for d, k in zip(toks, keep) if k]
    X = np.column_stack([treatment[keep], pid[keep]])
    return docs, X


def test_vignette_recovers_treatment_effect():
    docs, X = _load_and_prep()
    assert len(docs) > 300  # gadarian is 341 responses, a couple drop out

    model = STM(num_topics=3, seed=1)
    model.fit(docs, X, prevalence_names=["treatment", "pid_rep"], iters=80)

    # Outputs are well-formed.
    assert model.topic_word.shape == (3, len(model.vocabulary))
    assert np.allclose(model.doc_topic.sum(axis=1), 1.0)
    assert model.prevalence_effects.shape == (3, 2)  # (intercept,treatment,pid_rep) x (K-1)

    # The substantive finding: the treatment significantly shifts at least one
    # topic's prevalence (|z| > 1.96), and the effects across topics roughly
    # cancel (proportions sum to 1, so a positive shift implies a negative one).
    effects = stm.estimate_effect(model.doc_topic, X, feature_names=["treatment", "pid_rep"])
    tz = [e.z[e.feature_names.index("treatment")] for e in effects]
    assert max(abs(z) for z in tz) > 1.96, tz
    assert max(tz) > 1.96 and min(tz) < -1.96, tz  # one topic up, another down

    # Determinism.
    m2 = STM(num_topics=3, seed=1)
    m2.fit(docs, X, prevalence_names=["treatment", "pid_rep"], iters=80)
    assert np.array_equal(model.topic_word, m2.topic_word)
