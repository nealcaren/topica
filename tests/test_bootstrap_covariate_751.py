"""#751: bootstrap_stability resamples a per-document covariate design with the
same draw as the documents, so a covariate model (STM/DMR/...) can be bootstrapped
as itself instead of being silently downgraded to a covariate-free model.
"""

import numpy as np
import pytest

import topica
from topica.validation import _is_per_doc, _take_rows, bootstrap_stability


# --- unit: which kwargs are per-document, and row selection ------------------

def test_is_per_doc_distinguishes_design_from_names():
    D = 6
    X = np.arange(D * 2).reshape(D, 2)      # (D, p) design -> per-document
    names = ["a", "b"]                       # length p != D -> per-feature
    assert _is_per_doc(X, D) is True
    assert _is_per_doc(names, D) is False
    assert _is_per_doc("some string", D) is False
    assert _is_per_doc(500, D) is False       # a scalar fit-kwarg (iters)
    assert _is_per_doc(None, D) is False


def test_take_rows_numpy_pandas_list():
    pick = np.array([0, 0, 3, 5, 5, 1])
    X = np.arange(12).reshape(6, 2)
    assert np.array_equal(_take_rows(X, pick), X[pick])
    pd = pytest.importorskip("pandas")
    dfX = pd.DataFrame(X, columns=["c0", "c1"])
    assert np.array_equal(_take_rows(dfX, pick).to_numpy(), X[pick])
    content = [f"lab{i}" for i in range(6)]
    assert _take_rows(content, pick) == [content[i] for i in pick]


# --- alignment proof: the design row travels with its document ---------------

class _Recorder:
    """Minimal model that records the (docs, prevalence) each fit receives, so a
    test can prove the covariate rows were resampled in lock-step with the docs."""

    calls = []

    def __init__(self, num_topics=2):
        self.num_topics = num_topics
        self.vocabulary = None
        self.topic_word = None

    def fit(self, docs, **kw):
        _Recorder.calls.append((list(docs), kw.get("prevalence")))
        vocab = sorted({w for d in docs for w in d}) or ["_"]
        self.vocabulary = vocab
        self.topic_word = np.ones((self.num_topics, len(vocab)))
        return self


def test_design_rows_resample_in_lockstep_with_docs():
    # Doc i is tagged by a unique token "d{i}" and its design row encodes i, so a
    # correctly paired resample has prevalence[row] == index parsed from the token.
    _Recorder.calls.clear()
    D = 10
    docs = [[f"d{i}"] for i in range(D)]
    X = np.arange(D, dtype=float).reshape(D, 1)  # X[i] == i

    ref = _Recorder(2)
    ref.fit(docs, prevalence=X)
    _Recorder.calls.clear()  # drop the reference fit; keep only bootstrap fits

    bootstrap_stability(
        docs, reference=ref, n_boot=3, seed=0,
        model_factory=lambda s: _Recorder(2),
        prevalence=X,
    )
    assert _Recorder.calls, "no bootstrap fits were recorded"
    for docs_b, X_b in _Recorder.calls:
        assert X_b is not None and len(X_b) == D
        for row_docs, row_x in zip(docs_b, X_b):
            idx = int(row_docs[0][1:])       # parse i out of "d{i}"
            assert float(row_x[0]) == float(idx)  # design row matches its doc


def test_per_feature_names_passed_through_unresampled():
    _Recorder.calls.clear()
    D = 8
    docs = [[f"d{i}"] for i in range(D)]
    X = np.zeros((D, 2))
    names = ["intercept", "slope"]  # length 2 != D
    ref = _Recorder(2)
    ref.fit(docs, prevalence=X, prevalence_names=names)
    _Recorder.calls.clear()

    seen = {}

    def factory(s):
        m = _Recorder(2)
        orig = m.fit

        def fit(docs, **kw):
            seen["prevalence_names"] = kw.get("prevalence_names")
            return orig(docs, **kw)

        m.fit = fit
        return m

    bootstrap_stability(docs, reference=ref, n_boot=2, seed=0,
                        model_factory=factory, prevalence=X, prevalence_names=names)
    assert seen["prevalence_names"] == names  # untouched, not resampled


# --- integration: a real STM bootstraps as an STM ----------------------------

def test_stm_covariate_model_bootstraps_as_itself():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    A, B = ["tax", "market", "trade", "budget"], ["war", "troop", "army", "navy"]
    docs, party = [], []
    for i in range(60):
        lab = i % 2
        heavy, light = (A, B) if lab else (B, A)
        docs.append(rng.choice(heavy, 6).tolist() + rng.choice(light, 2).tolist())
        party.append("R" if lab else "D")
    X, names = topica.one_hot(pd.Series(party))

    stm = topica.STM(num_topics=2, seed=13).fit(docs, prevalence=X, prevalence_names=names)
    # Before #751 this raised "STM needs prevalence and/or content covariates".
    bs = topica.bootstrap_stability(
        docs, reference=stm, n_boot=4, seed=7,
        model_factory=lambda s: topica.STM(num_topics=2, seed=s),
        prevalence=X, prevalence_names=names,
    )
    assert isinstance(bs, topica.BootstrapStability)
    assert len(bs["stability"]) == 2
    assert np.all((bs["stability"] >= 0) & (bs["stability"] <= 1))


def test_plain_lda_path_unchanged():
    # No per-document kwargs -> behavior identical to before.
    docs = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"], ["c", "e", "f"]] * 8
    bs = topica.bootstrap_stability(docs, num_topics=3, n_boot=3, seed=1)
    assert len(bs["stability"]) == 3
