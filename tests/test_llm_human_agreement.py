"""llm.human_agreement: correlate an LLM metric with human ratings (Zheng et al. 2025
Fig. 2 / Stammbach et al. 2023). Purely numeric and deterministic -- no LLM call."""

import numpy as np
import pytest

import topica
import topica.llm as L
from topica.coherence import llm_human_agreement


def test_namespace_surface():
    assert callable(topica.llm.human_agreement)
    assert "human_agreement" in L.__all__
    assert not hasattr(topica, "llm_human_agreement")  # namespaced


def test_perfect_rank_agreement():
    llm = [1.0, 2.0, 3.0, 4.0, 5.0]
    human = [10, 20, 30, 40, 50]           # monotone -> Spearman 1.0
    r = llm_human_agreement(llm, human)
    assert r["method"] == "spearman"
    assert r["n"] == 5
    assert r["correlation"] == pytest.approx(1.0)
    assert r["pvalue"] <= 0.05


def test_spearman_is_rank_based_not_linear():
    llm = [1.0, 2.0, 3.0, 4.0]
    human = [1, 4, 9, 16]                   # monotone but nonlinear
    assert llm_human_agreement(llm, human, method="spearman")["correlation"] == pytest.approx(1.0)
    # pearson is < 1 on the same nonlinear pair
    assert llm_human_agreement(llm, human, method="pearson")["correlation"] < 1.0


def test_anti_correlation():
    r = llm_human_agreement([1, 2, 3, 4], [4, 3, 2, 1])
    assert r["correlation"] == pytest.approx(-1.0)


def test_nan_pairs_are_dropped():
    # an LLM rating that failed to parse comes back NaN and must be dropped, not crash.
    llm = [1.0, np.nan, 3.0, 4.0, 5.0]
    human = [1, 2, 3, 4, 5]
    r = llm_human_agreement(llm, human)
    assert r["n"] == 4                       # the NaN pair dropped
    assert r["correlation"] == pytest.approx(1.0)


def test_methods_supported():
    llm = [3.0, 1.0, 2.0, 5.0, 4.0]
    human = [3, 1, 2, 5, 4]
    for m in ("spearman", "pearson", "kendall"):
        assert llm_human_agreement(llm, human, method=m)["correlation"] == pytest.approx(1.0)


def test_errors():
    with pytest.raises(ValueError, match="same length"):
        llm_human_agreement([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="method"):
        llm_human_agreement([1, 2, 3], [1, 2, 3], method="bogus")
    with pytest.raises(ValueError, match="at least 2"):
        llm_human_agreement([1.0, np.nan], [np.nan, 2.0])  # only 0 usable pairs


def test_accepts_llm_coherence_style_array():
    # the intended call: feed the per-topic llm_coherence output + human ratings.
    llm = np.array([3.0, 3.0, 1.0, 2.0])     # e.g. from topica.llm.coherence(model, ...)
    human = np.array([2.8, 2.9, 1.2, 2.1])
    r = llm_human_agreement(llm, human)
    assert 0.0 < r["correlation"] <= 1.0 and r["n"] == 4
