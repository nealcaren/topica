"""Spectral init for content-covariate STM (issue #216).

Content models (STM with ``content=``) used a random base beta even under
``init="spectral"``, which is strongly multimodal: most seeds collapse to a
flat, no-group-content optimum. R ``stm`` initializes the base topics spectrally
even in the content (SAGE) model and starts the content deviations at zero, which
is deterministic and stable. These tests pin that behavior: a content fit is now
seed-independent and recovers non-degenerate group content.
"""

import numpy as np
import topica


def _group_content_corpus(seed=0):
    """Two topics worded differently by group (real group-content structure)."""
    rng = np.random.default_rng(seed)
    a0, b0 = ["cat", "dog", "pet"], ["feline", "canine", "pet"]
    a1, b1 = ["star", "moon", "sky"], ["nova", "lunar", "sky"]
    docs, groups = [], []
    for _ in range(80):
        docs.append(list(rng.choice(a0, 8))); groups.append("A")
        docs.append(list(rng.choice(a1, 8))); groups.append("A")
        docs.append(list(rng.choice(b0, 8))); groups.append("B")
        docs.append(list(rng.choice(b1, 8))); groups.append("B")
    return docs, groups


def _fit(seed, docs, groups):
    m = topica.models.STM(num_topics=2, seed=seed)  # init defaults to spectral
    m.fit(docs, content=groups, iters=80)
    return m


def test_stm_content_spectral_is_seed_independent():
    docs, groups = _group_content_corpus()
    m1, m2 = _fit(1, docs, groups), _fit(2, docs, groups)
    diff = np.abs(np.asarray(m1.topic_word) - np.asarray(m2.topic_word)).max()
    # Spectral base + zero content kappa + deterministic variational EM => the
    # fit no longer depends on the seed.
    assert diff < 1e-9, f"content fit not seed-independent (max|diff|={diff:.2e})"


def test_stm_content_recovers_group_differences():
    docs, groups = _group_content_corpus()
    m = _fit(1, docs, groups)
    twg = np.asarray(m.topic_word_by_group)  # K x G x V
    ga, gb = m.groups.index("A"), m.groups.index("B")
    sep = np.abs(twg[:, ga, :] - twg[:, gb, :]).mean()
    # The structured optimum (groups word topics differently) is found, not the
    # collapsed flat-content optimum (sep ~ 0).
    assert sep > 0.02, f"group content collapsed (separation={sep:.4f})"
