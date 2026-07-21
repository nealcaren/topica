"""STM content_time reading layer: content_trajectory / content_divergence (#365).

Ports faSTM's readers to the STM `content_time` surface. These tests validate
recovery of a known Dem/Rep wording drift, the anchor-word topic realignment, the
Hellinger/TV distances, and the design-preserving bootstrap CIs (the #340 fix).
"""

import numpy as np
import pytest

import topica
from topica import content


def _drift_corpus(per_cell=60, seed=0):
    """Two groups x four periods. One topic drifts: 'climate' shifts from Rep to
    Dem and 'border' the other way, crossing over near the middle; 'econ/jobs/tax'
    is shared filler. Returns (docs, groups, periods, period_labels)."""
    periods = [1990, 2000, 2010, 2020]
    docs, grp, per = [], [], []
    for pi, yr in enumerate(periods):
        d = pi / (len(periods) - 1)  # 0 -> 1
        for _ in range(per_cell):
            docs.append(["econ", "jobs", "tax"] * 2
                        + ["climate"] * int(1 + 4 * d) + ["border"] * int(1 + 4 * (1 - d)))
            grp.append("Dem"); per.append(yr)
            docs.append(["econ", "jobs", "tax"] * 2
                        + ["border"] * int(1 + 4 * d) + ["climate"] * int(1 + 4 * (1 - d)))
            grp.append("Rep"); per.append(yr)
    return docs, grp, per, periods


@pytest.fixture(scope="module")
def fit():
    docs, grp, per, periods = _drift_corpus()
    m = topica.STM(num_topics=3, seed=1)
    m.fit(docs, content=grp, content_time=per, content_prior="l1", iters=150)
    return m, docs, grp, per


ANCHOR = ["climate", "border", "econ"]


def test_trajectory_recovers_drift(fit):
    m = fit[0]
    tr = content.content_trajectory(m, ["climate", "border"], groups=("Dem", "Rep"),
                                    anchor_words=ANCHOR)
    assert tr.periods == ["1990", "2000", "2010", "2020"]
    clim = tr.estimate[tr.words.index("climate")]
    bord = tr.estimate[tr.words.index("border")]
    # Dem-Rep climate contrast rises monotonically across periods and flips sign;
    # border is its mirror image.
    assert np.all(np.diff(clim) > 0)
    assert clim[0] < 0 < clim[-1]
    assert np.allclose(clim, -bord, atol=1e-9)


def test_divergence_peaks_at_the_extremes(fit):
    m = fit[0]
    dv = content.content_divergence(m, groups=("Dem", "Rep"), anchor_words=ANCHOR)
    d = dv.divergence
    # groups word the topic most differently at the endpoints, alike in the middle
    assert d[0] > d[1] and d[-1] > d[-2]
    assert dv.measure == "hellinger"
    # TV is a different but also non-negative distance
    tv = content.content_divergence(m, groups=("Dem", "Rep"), anchor_words=ANCHOR, measure="tv")
    assert np.all(tv.divergence >= 0)


def test_anchor_realignment_matches_explicit_topic(fit):
    m = fit[0]
    # The anchor-selected topic equals the explicit index for that topic.
    by_anchor = content.content_trajectory(m, ["climate"], anchor_words=ANCHOR)
    by_topic = content.content_trajectory(m, ["climate"], topic=by_anchor.topic)
    assert np.allclose(by_anchor.estimate, by_topic.estimate, equal_nan=True)


def test_defaults_first_two_groups(fit):
    m = fit[0]
    tr = content.content_trajectory(m, ["climate"], topic=0)
    assert tr.groups == ("Dem", "Rep")


def test_bootstrap_ci_brackets_point_estimate(fit):
    m, docs, grp, per = fit
    fk = dict(num_topics=3, seed=1, content=grp, content_time=per,
              content_prior="l1", iters=150)
    dv = content.content_divergence(m, groups=("Dem", "Rep"), anchor_words=ANCHOR,
                                    ci=True, corpus=docs, fit_kwargs=fk, B=8, seed=3)
    assert dv.ci_low is not None and dv.ci_high is not None
    assert np.all(dv.ci_low <= dv.ci_high + 1e-9)
    # the full-data estimate sits within (or at) the resampling band
    assert np.all(dv.ci_low - 1e-6 <= dv.divergence)
    assert np.all(dv.divergence <= dv.ci_high + 1e-6)


def test_cluster_bootstrap_runs(fit):
    m, docs, grp, per = fit
    fk = dict(num_topics=3, seed=1, content=grp, content_time=per,
              content_prior="l1", iters=120)
    # cluster on (group, period) cells — whole cells resampled together
    cluster = [f"{g}:{p}" for g, p in zip(grp, per)]
    tr = content.content_trajectory(m, ["climate"], groups=("Dem", "Rep"),
                                    anchor_words=ANCHOR, ci=True, corpus=docs,
                                    fit_kwargs=fk, cluster=cluster, B=6, seed=5)
    assert tr.ci_low is not None and tr.ci_low.shape == tr.estimate.shape


def test_ci_requires_anchor_and_corpus(fit):
    m, docs, grp, per = fit
    fk = dict(num_topics=3, seed=1, content=grp, content_time=per, iters=100)
    with pytest.raises(ValueError, match="anchor_words is required"):
        content.content_divergence(m, topic=0, ci=True, corpus=docs, fit_kwargs=fk, B=3)
    with pytest.raises(ValueError, match="corpus"):
        content.content_divergence(m, anchor_words=ANCHOR, ci=True, fit_kwargs=fk, B=3)


def test_non_content_time_model_errors():
    docs = [["a", "b", "c"]] * 20 + [["x", "y", "z"]] * 20
    m = topica.STM(num_topics=2, seed=1)
    m.fit(docs, content=["p"] * 20 + ["q"] * 20, iters=50)  # content but no content_time
    with pytest.raises(ValueError, match="content_time"):
        content.content_trajectory(m, ["a"], topic=0)


def test_invalid_measure(fit):
    with pytest.raises(ValueError, match="hellinger"):
        content.content_divergence(fit[0], anchor_words=ANCHOR, measure="kl")


def test_to_frame(fit):
    tr = content.content_trajectory(fit[0], ["climate"], anchor_words=ANCHOR)
    df = tr.to_frame()
    assert list(df.columns) == ["word", "period", "estimate"]
    assert len(df) == 4  # 1 word x 4 periods
