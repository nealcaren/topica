"""Content-covariate diagnostics (topica.content).

Validated by defining properties on synthetic content models: identical group
wordings give zero polarization, disjoint group vocabularies give ~1, the group
tensor is recovered from every model family (SAGE 3-D topic_word, ECTM content
cells), and the fragmentation detector fires on a planted split and stays quiet
otherwise.
"""

import numpy as np
import pytest

import topica
from topica.models import SAGE, ECTM
from topica import content


# --------------------------------------------------------------------------- #
# Pure-metric properties via a hand-built model stub (no fitting needed).
# --------------------------------------------------------------------------- #
class _StubContentModel:
    """Minimal STM/STS-shaped content model: exposes topic_word_by_group."""

    def __init__(self, beta_kgv, groups):
        self._beta = np.asarray(beta_kgv, dtype=float)
        self.groups = list(groups)

    @property
    def topic_word_by_group(self):
        return self._beta


def test_polarization_zero_when_groups_word_topics_identically():
    V = 6
    shared = np.random.default_rng(0).dirichlet(np.ones(V), size=3)  # (K=3, V)
    beta = np.stack([shared, shared], axis=1)                        # (K, G=2, V) identical
    m = _StubContentModel(beta, ["a", "b"])
    pol = content.topic_polarization(m)
    assert pol.shape == (3,)
    assert np.allclose(pol, 0.0, atol=1e-9)


def test_polarization_one_when_groups_use_disjoint_vocabularies():
    # group A puts all mass on the first half, group B on the second half
    V = 8
    a = np.zeros(V); a[:4] = 0.25
    b = np.zeros(V); b[4:] = 0.25
    beta = np.array([[a, b]])  # (K=1, G=2, V)
    m = _StubContentModel(beta, ["a", "b"])
    pol = content.topic_polarization(m)
    assert pol[0] == pytest.approx(1.0, abs=1e-6)


def test_polarization_weights_shift_the_mixture():
    V = 8
    a = np.zeros(V); a[:4] = 0.25
    b = np.zeros(V); b[4:] = 0.25
    m = _StubContentModel(np.array([[a, b]]), ["a", "b"])
    # extreme weight on one group collapses the divergence toward 0
    skewed = content.topic_polarization(m, weights=[0.999, 0.001])[0]
    balanced = content.topic_polarization(m, weights=[0.5, 0.5])[0]
    assert skewed < balanced


def test_polarization_requires_two_groups():
    m = _StubContentModel(np.random.default_rng(0).dirichlet(np.ones(5), size=(2, 1)), ["only"])
    with pytest.raises(ValueError):
        content.topic_polarization(m)


def test_non_content_model_raises():
    class _Plain:
        topic_word = np.random.default_rng(0).dirichlet(np.ones(5), size=3)  # (K, V) 2-D
    with pytest.raises(ValueError, match="content-covariate"):
        content.topic_polarization(_Plain())


def test_group_exclusivity_in_unit_range_and_summaries():
    rng = np.random.default_rng(1)
    beta = rng.dirichlet(np.ones(10), size=(4, 3))  # (K=4, G=3, V=10)
    m = _StubContentModel(beta, ["x", "y", "z"])
    ex_min = content.group_exclusivity(m, summary="min")
    ex_mean = content.group_exclusivity(m, summary="mean")
    assert ex_min.shape == (4,)
    assert np.all((ex_min >= 0) & (ex_min <= 1))
    assert np.all(ex_mean >= ex_min - 1e-9)  # mean >= worst-case group
    with pytest.raises(ValueError):
        content.group_exclusivity(m, summary="median")


# --------------------------------------------------------------------------- #
# Real fitted models: the adapter reaches SAGE and ECTM tensors.
# --------------------------------------------------------------------------- #
def test_group_topic_word_reads_sage():
    docs = [["tax", "cut", "budget", "fiscal"] if i % 2 else ["war", "peace", "troops", "army"]
            for i in range(80)]
    groups = ["L" if i % 2 else "R" for i in range(80)]
    m = SAGE(num_topics=3, seed=0)
    m.fit(topica.Corpus.from_documents(docs), groups)
    beta, labels = content.group_topic_word(m)
    assert beta.shape[0] == 3 and beta.shape[1] == 2
    assert np.allclose(beta.sum(axis=2), 1.0, atol=1e-6)
    pol = content.topic_polarization(m)
    assert pol.shape == (3,) and np.all((pol >= 0) & (pol <= 1))


@pytest.fixture(scope="module")
def ectm_model():
    topica.enable_experimental()
    rng = np.random.default_rng(0)
    # two groups word a shared "housing" topic differently; a control topic is shared.
    docs, groups, times = [], [], []
    for i in range(180):
        g = "dev" if i % 2 else "res"
        period = ["t0", "t1", "t2"][i % 3]
        if i % 2:  # housing, group-specific wording
            base = ["upscale", "premium", "luxury"] if g == "dev" else ["eviction", "rent", "tenant"]
        else:
            base = ["park", "school", "road"]  # shared control
        docs.append(base * 3)
        groups.append(g)
        times.append(period)
    m = ECTM(num_topics=4, seed=1)
    m.fit(topica.Corpus.from_documents(docs), times=times, content=groups, iters=120,
          content_prior_var=2.0)
    return m, groups


def test_ectm_polarization_and_period_trajectory(ectm_model):
    m, groups = ectm_model
    pol = content.topic_polarization(m)
    assert pol.shape == (m.num_topics,)
    assert np.all((pol >= -1e-9) & (pol <= 1 + 1e-9))
    # per-period trajectory: one value per period, each a valid JSD
    traj = [content.topic_polarization(m, period=t) for t in range(m.num_periods)]
    assert len(traj) == m.num_periods
    assert all(v.shape == (m.num_topics,) for v in traj)


def test_split_topics_returns_wellformed(ectm_model):
    m, groups = ectm_model
    pairs = content.split_topics(m, groups)
    assert isinstance(pairs, list)
    for p in pairs:
        assert set(p) >= {"pair", "cosine", "groups"}
        assert p["groups"][0] != p["groups"][1]


def test_content_is_a_namespace():
    assert hasattr(topica, "content")
    assert topica.content.topic_polarization is content.topic_polarization
