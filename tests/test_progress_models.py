"""Live fit-progress callback across the slow high-value models (#786 phase 2).

Each model's fit(progress=cb) must call cb(iteration, total, info) with a dict
info, drive iteration up to total (for the ones that run a fixed budget), and not
change the fit. LDA/DMR/LabeledLDA/SAGE/GSDMM/keyATM/HDP are covered elsewhere
(phase 1); this pins the twelve wired in phase 2.
"""

import numpy as np
import pytest

import topica

DOCS = [["tax", "budget", "vote", "tax", "bill"]] * 30 + [
    ["health", "care", "clinic", "care", "doctor"]
] * 30
VOCAB = sorted({w for d in DOCS for w in d})
EMB = np.random.RandomState(0).randn(len(VOCAB), 8)
DOC_EMB = np.random.RandomState(1).randn(len(DOCS), 16)
COV = np.array([[0.0]] * 30 + [[1.0]] * 30)
TIMES = [0] * 30 + [1] * 30


def _fit(model_name, cb):
    """Fit one model with progress=cb at a small iters; return (model, iters)."""
    if model_name == "CTM":
        return topica.CTM(3, seed=13).fit(DOCS, iters=12, progress=cb), 12
    if model_name == "STM":
        return topica.STM(3, seed=13).fit(DOCS, COV, iters=12, progress=cb), 12
    if model_name == "DTM":
        # DTM early-stops on its bound, so total is the cap, not necessarily reached.
        return topica.DTM(3, seed=13).fit(DOCS, TIMES, iters=10, progress=cb), 10
    if model_name == "ProdLDA":
        return topica.ProdLDA(3, seed=13).fit(DOCS, iters=12, progress=cb), 12
    if model_name == "ETM":
        return topica.ETM(3, seed=13).fit(DOCS, EMB, VOCAB, iters=12, progress=cb), 12
    if model_name == "DETM":
        return (
            topica.DETM(3, seed=13).fit(
                DOCS, EMB, VOCAB, times=TIMES, iters=8, progress=cb
            ),
            8,
        )
    if model_name == "Scholar":
        return (
            topica.Scholar(3, seed=13).fit(
                DOCS, covariates=COV, iters=12, progress=cb
            ),
            12,
        )
    if model_name == "FASTopic":
        return topica.FASTopic(3, seed=13).fit(DOCS, DOC_EMB, iters=12, progress=cb), 12
    if model_name == "InfoCTM":
        docs_b = [["impuesto", "voto"]] * 30 + [["salud", "clinica"]] * 30
        dic = [("tax", "impuesto"), ("health", "salud")]
        return (
            topica.InfoCTM(num_topics=3, seed=13).fit(
                DOCS, docs_b, dictionary=dic, iters=15, batch_size=8, progress=cb
            ),
            15,
        )
    if model_name == "SeededLDA":
        seeds = {"fiscal": ["tax", "budget"], "health": ["health", "care"]}
        return (
            topica.SeededLDA(seeds, seed=13).fit(
                DOCS, iters=60, check_every=10, progress=cb
            ),
            60,
        )
    if model_name == "BTM":
        return topica.BTM(3, seed=13).fit(DOCS, iters=40, progress=cb), 40
    if model_name == "HLDA":
        return topica.HLDA(seed=13).fit(DOCS, iters=30, progress=cb), 30
    raise AssertionError(model_name)


# ll-reporting models (info carries a real metric) vs bare (empty info dict).
LL_MODELS = [
    "CTM", "STM", "DTM", "ProdLDA", "ETM", "DETM",
    "Scholar", "FASTopic", "InfoCTM", "SeededLDA",
]
BARE_MODELS = ["BTM", "HLDA"]


@pytest.mark.parametrize("model_name", LL_MODELS + BARE_MODELS)
def test_progress_fires_with_iteration_total_info(model_name):
    calls = []
    with _no_warn():
        _fit(model_name, lambda it, total, info: calls.append((it, total, dict(info))))
    assert calls, f"{model_name}: progress never fired"
    its, totals, infos = zip(*calls)
    assert all(isinstance(i, int) and isinstance(t, int) for i, t, _ in calls)
    assert all(isinstance(inf, dict) for inf in infos)
    # iteration is 1-based, monotonically non-decreasing, never past its own total.
    assert its[0] >= 1 and all(a <= b for a, b in zip(its, its[1:]))
    assert all(i <= t for i, t in zip(its, totals))
    # In-progress frames report the iteration budget as total; on early
    # convergence the fit emits a final snap frame with iteration == total (100%),
    # so the last frame always closes the bar even when it stopped short of budget.
    assert its[-1] == totals[-1], f"{model_name}: bar did not close at 100%"
    if model_name in LL_MODELS:
        assert all("ll" in inf for inf in infos), f"{model_name}: missing ll metric"
    else:
        assert all(inf == {} for inf in infos), f"{model_name}: expected bare info"


@pytest.mark.parametrize(
    "model_name", ["CTM", "ProdLDA", "SeededLDA", "BTM", "FASTopic"]
)
def test_progress_does_not_change_the_fit(model_name):
    m_a, _ = _fit(model_name, None)
    m_b, _ = _fit(model_name, lambda *xs: None)

    def topic_word(m):
        tw = m.topic_word
        return np.asarray(tw() if callable(tw) else tw)

    assert np.allclose(topic_word(m_a), topic_word(m_b))


class _no_warn:
    """Context manager that silences warnings (some fits warn on tiny corpora)."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore")
        return self

    def __exit__(self, *a):
        return self._cm.__exit__(*a)
