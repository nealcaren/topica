"""Behavioral conformance sweep (issue #117).

The interface conformance check (``tests/test_conformance.py``) verifies that
every estimator *exposes* the contract attributes. This file verifies the
contract actually *works end to end*: for every transform-capable model we run
the full fit -> save -> load -> transform workflow and assert that

  1. ``transform`` on held-out documents returns a valid ``(n, K)`` matrix,
  2. ``save``/``load`` preserves ``topic_word`` bit-for-bit, and
  3. ``transform`` on the loaded model reproduces the pre-save result exactly
     (the round-trip is behaviorally identical, not just structurally present).

This is the structural cover the audit (#117) asked for: it exercises the
save-format + transform interaction across the whole model family in one place,
so a model that silently drops state on load (the #102 failure mode) is caught
for every model, not just the few with a bespoke round-trip test.

Models without a flat held-out ``transform`` (HLDA, DTM, GSDMM) and the
embedding-input models (ETM, FASTopic, BERTopic, Top2Vec), whose fit/transform
take user-supplied vectors, are covered elsewhere and excluded here.
"""

import numpy as np
import pytest

import topica

# A small, well-separated two-theme corpus: stable topics at low iteration counts.
DOCS = (
    [["cat", "dog", "pet", "vet"]] * 15
    + [["star", "moon", "sky", "sun"]] * 15
    + [["cat", "dog", "star", "moon"]] * 10
)
HELD_OUT = [["cat", "dog"], ["star", "moon"], ["cat", "star"]]
N = len(DOCS)


# ---------------------------------------------------------------------------
# Per-model fitted-model builders. Each returns a freshly fitted model; the
# fit inputs are the minimal ones each fit() requires (covariates, labels,
# groups, sentiment seed), all derived from the shared corpus above.
# ---------------------------------------------------------------------------

def _lda():
    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=150)
    return m


def _dmr():
    X = np.ones((N, 1))
    m = topica.models.DMR(num_topics=3, seed=1)
    m.fit(DOCS, X, feature_names=["x"], iters=80)
    return m


def _sage():
    groups = ["g0", "g1"] * (N // 2)
    m = topica.models.SAGE(num_topics=3, seed=1)
    m.fit(DOCS, groups, iters=80)
    return m


def _pa():
    m = topica.models.PA(num_super=2, num_sub=3, seed=1)
    m.fit(DOCS, iters=150)
    return m


def _pt():
    m = topica.models.PT(num_topics=3, num_pseudo=10, seed=1)
    m.fit(DOCS, iters=150)
    return m


def _hdp():
    m = topica.models.HDP(seed=1)
    m.fit(DOCS, iters=150)
    return m


def _labeled():
    m = topica.models.LabeledLDA(seed=1)
    m.fit(DOCS, [["t0", "t1"]] * N, iters=80)
    return m


def _supervised():
    y = np.array([0.0, 1.0] * (N // 2))
    m = topica.models.SupervisedLDA(num_topics=3, seed=1)
    m.fit(DOCS, y, iters=80)
    return m


def _keyatm():
    m = topica.models.KeyATM({"animals": ["cat", "dog"], "space": ["star", "moon"]},
                      num_topics=3, seed=1)
    m.fit(DOCS, iters=80)
    return m


def _seeded():
    m = topica.models.SeededLDA({"animals": ["cat", "dog"], "space": ["star", "moon"]},
                         residual=1, seed=1)
    m.fit(DOCS, iters=150)
    return m


def _stm():
    X = np.ones((N, 1))
    m = topica.models.STM(num_topics=3, seed=1)
    m.fit(DOCS, X, prevalence_names=["x"], iters=40)
    return m


def _ctm():
    m = topica.models.CTM(num_topics=3, seed=1)
    m.fit(DOCS, iters=40)
    return m


def _sts():
    seed_vals = [0.0, 1.0] * (N // 2)
    m = topica.models.STS(num_topics=3, seed=1)
    m.fit(DOCS, sentiment_seed=seed_vals, iters=30)
    return m


def _prodlda():
    m = topica.models.ProdLDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=60)
    return m


BUILDERS = {
    "LDA": _lda,
    "DMR": _dmr,
    "SAGE": _sage,
    "PA": _pa,
    "PT": _pt,
    "HDP": _hdp,
    "LabeledLDA": _labeled,
    "SupervisedLDA": _supervised,
    "KeyATM": _keyatm,
    "SeededLDA": _seeded,
    "STM": _stm,
    "CTM": _ctm,
    "STS": _sts,
    "ProdLDA": _prodlda,
}


@pytest.mark.parametrize("name", list(BUILDERS), ids=list(BUILDERS))
def test_fit_save_load_transform(name, tmp_path):
    """fit -> save -> load -> transform is behaviorally identical for every model."""
    model = BUILDERS[name]()
    k = model.num_topics
    assert k >= 1

    # 1. transform on held-out docs is a valid (n, K) simplex.
    before = np.asarray(model.transform(HELD_OUT))
    assert before.shape == (len(HELD_OUT), k), (
        f"{name}: transform shape {before.shape} != ({len(HELD_OUT)}, {k})"
    )
    assert np.isfinite(before).all(), f"{name}: transform produced non-finite values"
    np.testing.assert_allclose(
        before.sum(axis=1), np.ones(len(HELD_OUT)), atol=1e-4,
        err_msg=f"{name}: transform rows are not a probability simplex",
    )

    # 2. save/load preserves topic_word bit-for-bit.
    path = str(tmp_path / f"{name}.tt")
    model.save(path)
    loaded = type(model).load(path)
    np.testing.assert_array_equal(
        np.asarray(model.topic_word), np.asarray(loaded.topic_word),
        err_msg=f"{name}: topic_word changed across save/load",
    )
    assert loaded.num_topics == k, f"{name}: num_topics changed across save/load"

    # 3. transform on the loaded model reproduces the pre-save result exactly.
    after = np.asarray(loaded.transform(HELD_OUT))
    np.testing.assert_array_equal(
        before, after,
        err_msg=f"{name}: transform differs after save/load — state dropped on load",
    )


# ---------------------------------------------------------------------------
# Targeted gaps the audit (#117) named individually.
# ---------------------------------------------------------------------------

def test_multithreaded_fit_save_load_roundtrip(tmp_path):
    """save/load after a multithreaded (AD-LDA) fit — the parallelism +
    serialization interaction the paper claims, untested until now."""
    m = topica.models.LDA(num_topics=3, seed=1)
    m.fit(DOCS, iters=150, num_threads=4)
    before = np.asarray(m.transform(HELD_OUT))

    path = str(tmp_path / "mt.tt")
    m.save(path)
    loaded = topica.models.LDA.load(path)

    np.testing.assert_array_equal(
        np.asarray(m.topic_word), np.asarray(loaded.topic_word),
        err_msg="multithreaded fit: topic_word changed across save/load",
    )
    np.testing.assert_array_equal(
        before, np.asarray(loaded.transform(HELD_OUT)),
        err_msg="multithreaded fit: transform differs after save/load",
    )


@pytest.mark.parametrize("sampler", ["sparse", "warp", "cvb0"])
def test_sampler_backend_survives_load(sampler, tmp_path):
    """The chosen sampler backend (and the state it needs) must survive a
    save/load so the loaded model stays behaviorally identical (#102b). Each
    backend has its own doc-phase; if the flag were dropped on load, transform
    would fall back to the default sampler and diverge."""
    m = topica.models.LDA(num_topics=3, seed=1, sampler=sampler)
    m.fit(DOCS, iters=150)
    before = np.asarray(m.transform(HELD_OUT))

    path = str(tmp_path / f"{sampler}.tt")
    m.save(path)
    loaded = topica.models.LDA.load(path)

    np.testing.assert_array_equal(
        np.asarray(m.topic_word), np.asarray(loaded.topic_word),
        err_msg=f"sampler={sampler}: topic_word changed across save/load",
    )
    np.testing.assert_array_equal(
        before, np.asarray(loaded.transform(HELD_OUT)),
        err_msg=f"sampler={sampler}: transform diverged after load (backend flag lost?)",
    )


def test_svi_honors_convergence_tol():
    """CTM's stochastic variational inference (inference="svi") must respect
    convergence_tol and early-stop, not run the full iters every time."""
    iters = 100
    m = topica.models.CTM(num_topics=3, seed=1)
    m.fit(DOCS, iters=iters, inference="svi", convergence_tol=1e-2)

    assert m.converged is True, "loose convergence_tol should early-stop SVI"
    assert 0 < len(m.fit_history) < iters, (
        f"SVI should stop before {iters} epochs under a loose tol; "
        f"ran {len(m.fit_history)}"
    )
    # The fit is still valid: a proper topic-word simplex.
    tw = np.asarray(m.topic_word)
    np.testing.assert_allclose(tw.sum(axis=1), np.ones(tw.shape[0]), atol=1e-4)
