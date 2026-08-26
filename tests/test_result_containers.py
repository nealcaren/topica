"""Conformance for the diagnostic result containers.

Every helper that returns a *container* of results should present itself the same
way: a ``.to_frame()`` a pandas user reaches for, and — for the ``list``-backed
ones — a ``__repr__`` that summarizes instead of dumping raw rows. Two of these
(``predicted_prevalence`` returning a bare list, ``SearchKResult`` printing raw
per-K dicts) slipped because nothing enforced the contract; this test is that
enforcement (issue #817, from the #816 sample-user audit).
"""
import numpy as np
import pytest

import topica
from topica._results import FrameDict
from topica.select import SearchKResult
from topica.stm import EffectList, PredictedPrevalenceList

# The list-backed result containers. A bare `list` return (predicted_prevalence's
# old shape) or an un-overridden repr (SearchKResult's old shape) is the bug.
LIST_CONTAINERS = [EffectList, SearchKResult, PredictedPrevalenceList]


def _all_subclasses(cls):
    out = set(cls.__subclasses__())
    for c in list(out):
        out |= _all_subclasses(c)
    return out


@pytest.mark.parametrize("cls", LIST_CONTAINERS, ids=lambda c: c.__name__)
def test_list_container_has_to_frame_and_custom_repr(cls):
    assert callable(getattr(cls, "to_frame", None)), f"{cls.__name__} needs .to_frame()"
    assert cls.__repr__ is not list.__repr__, (
        f"{cls.__name__} must override __repr__ — the raw list dump is unreadable")


def test_framedict_containers_all_tabulate():
    subs = _all_subclasses(FrameDict)
    assert subs, "expected FrameDict result containers"
    for cls in subs:
        assert callable(getattr(cls, "to_frame", None)), f"{cls.__name__} needs .to_frame()"


def test_containers_render_and_tabulate_end_to_end():
    # A real fit: the containers must summarize (not dump) and tabulate.
    df = topica.datasets.load_gadarian()
    c = topica.from_dataframe(
        df, text_col="open.ended.response", metadata_cols=["treatment", "pid_rep"],
        stopwords=topica.stopwords("english"), min_length=3, min_doc_freq=3,
    )
    X = np.column_stack([c.metadata[k].to_numpy(float) for k in ["treatment", "pid_rep"]])

    res = topica.select.search_k(c, ks=[3, 4], seed=13)
    assert isinstance(res, SearchKResult)
    r = repr(res)
    assert r.startswith("SearchKResult:") and "best_k()" in r  # adjudicates, not a dict dump
    assert not r.lstrip().startswith("[{")
    assert res.to_frame().shape[0] == 2

    m = topica.STM(num_topics=3, seed=13).fit(
        c, prevalence=X, prevalence_names=["treatment", "pid_rep"], iters=40)
    pp = topica.effects.predicted_prevalence(
        m, X=X, feature_names=["treatment", "pid_rep"], contrast={"treatment": [0, 1]}, corpus=c)
    assert isinstance(pp, PredictedPrevalenceList)
    assert repr(pp).startswith("PredictedPrevalenceList:")
    assert pp.to_frame().shape[0] == 3  # one row per topic (single contrast point)
    assert isinstance(pp, list) and pp[0].topic == 0  # still a plain list underneath
