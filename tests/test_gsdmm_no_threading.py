"""GSDMM intentionally does NOT support AD-LDA threading (#566).

Unlike the sibling collapsed-Gibbs count models (DMR, LabeledLDA, SeededLDA, BTM),
GSDMM's Movie Group Process discovers the cluster count K through within-sweep
reinforcement — once a document births a cluster, later documents in the same
sweep join it, driving the collapse. Approximate-parallel (AD-LDA) sampling would
break that reinforcement and make K depend on the thread count, which is
unacceptable for a headline structural output. GSDMM therefore accepts only
num_threads=1 and rejects num_threads>1 with a clear ValueError (rather than a
bare TypeError) so users arriving from the sibling models get an explanation.
"""

import pytest

import topica

DOCS = [["a", "b", "c"], ["a", "b"], ["x", "y", "z"], ["x", "y"]] * 20


def test_num_threads_1_is_accepted():
    # Default and explicit num_threads=1 both fit.
    topica.GSDMM(6, seed=1).fit(DOCS, iters=20)
    topica.GSDMM(6, seed=1, num_threads=1).fit(DOCS, iters=20, num_threads=1)


def test_settings_reports_num_threads_1():
    assert topica.GSDMM(6).settings["num_threads"] == 1


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_constructor_rejects_multithreading(nt):
    with pytest.raises(ValueError, match="sequentially"):
        topica.GSDMM(6, num_threads=nt)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fit_rejects_multithreading(nt):
    with pytest.raises(ValueError, match="sequentially"):
        topica.GSDMM(6).fit(DOCS, num_threads=nt)


def test_error_message_points_to_sibling_models():
    with pytest.raises(ValueError, match="DMR, LabeledLDA, SeededLDA, BTM"):
        topica.GSDMM(6, num_threads=2)
