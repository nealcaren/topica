"""Uniform convergence interface tests (issue #46 Parts A and B).

Every topica estimator must expose:
  - fit_history: list[(int, float)] -- (iteration, objective) pairs
  - converged: bool | None -- True if early-stop fired; None for cluster models

Tier 0 class-level presence is verified by test_conformance.py.  This file
does the fitted-model checks: correct types, non-empty trace where expected,
and early-stop semantics.
"""

from __future__ import annotations

import pytest

import topica
from topica.conformance import REGISTRY

# ---------------------------------------------------------------------------
# Toy corpus (two clusters, enough to fit quickly)
# ---------------------------------------------------------------------------

_ANIMAL = ["cat", "dog", "fish", "cat", "dog"]
_SPACE  = ["planet", "star", "moon", "rocket", "planet"]
_TOY    = [list(_ANIMAL) for _ in range(15)] + [list(_SPACE) for _ in range(15)]

# Cluster models: fit_history == [] and converged is None by design (not a gap).
_CLUSTER_MODELS = {"BERTopic", "Top2Vec"}

# Direct (non-iterative) solves: LSA is a one-shot truncated SVD, so like the
# cluster models it has no per-iteration trace (fit_history == []) and no
# convergence state (converged is None). It takes no `iters` argument.
_DIRECT_SOLVE_MODELS = {"LSA"}

# Models whose fitted state has no iterative objective trace and no early-stop
# convergence flag: the cluster models plus the direct SVD solve.
_NO_TRACE_MODELS = _CLUSTER_MODELS | _DIRECT_SOLVE_MODELS

# Models that intentionally never early-stop, so converged is always False.
# HDP and GSDMM discover their topic/cluster counts, so a log-likelihood plateau
# is not a convergence signal; DTM and HLDA expose no flat per-iteration objective.
# (keyATM is NOT here: it supports opt-in convergence_tol early-stopping.)
_CONVERGED_FALSE_ALWAYS = {"HDP", "GSDMM", "DTM", "HLDA"}

# ---------------------------------------------------------------------------
# Helpers to build a minimal fitted instance for any registry model
# ---------------------------------------------------------------------------

def _fit_model(name: str, factory):
    """Return a fitted instance for model *name* using the toy corpus."""
    import numpy as np

    model = factory()

    # DTM: requires positional `times` (time-slice per document)
    if name == "DTM":
        times = [0] * 15 + [1] * 15
        model.fit(_TOY, times, iters=5)
        return model

    # LabeledLDA: `labels` is list[list[str]] (one label-list per document)
    if name == "LabeledLDA":
        labels = [["animal"]] * 15 + [["space"]] * 15
        model.fit(_TOY, labels, iters=20, check_every=10)
        return model

    # SupervisedLDA: positional `y` is a per-document real-valued response
    if name == "SupervisedLDA":
        y = [0.0] * 15 + [1.0] * 15
        model.fit(_TOY, y, iters=10, check_every=1)
        return model

    # DMR: positional `features` is a numeric (num_docs, F) covariate matrix
    if name == "DMR":
        features = np.array([[1.0, 0.0]] * 15 + [[0.0, 1.0]] * 15)
        model.fit(_TOY, features, iters=20, check_every=10)
        return model

    # SAGE: positional `groups` is a per-document group label
    if name == "SAGE":
        groups = ["animal"] * 15 + ["space"] * 15
        model.fit(_TOY, groups, iters=20, check_every=10)
        return model

    # STM: at minimum supply a prevalence covariate (else it errors)
    if name == "STM":
        prevalence = np.array([[0.0]] * 15 + [[1.0]] * 15)
        model.fit(_TOY, prevalence, iters=5)
        return model

    # STS: requires a per-document sentiment_seed (the aggregation groups)
    if name == "STS":
        sent = [float(i % 3) for i in range(len(_TOY))]
        model.fit(_TOY, sent, iters=5)
        return model

    # ETM: requires positional word_embeddings and vocabulary
    if name == "ETM":
        import topica
        # Build a minimal vocabulary from the toy corpus
        vocab = list({w for doc in _TOY for w in doc})
        rng = np.random.default_rng(42)
        word_emb = rng.standard_normal((len(vocab), 8))
        model.fit(_TOY, word_emb, vocab, iters=5)
        return model

    # HLDA does not take num_topics at construct time; fits with default iters
    if name == "HLDA":
        model.fit(_TOY, iters=5)
        return model

    # Contextualized ProdLDA (CombinedTM, ZeroShotTM): require doc_embeddings
    # as a (num_docs, E) array alongside the tokens.
    if name in ("CombinedTM", "ZeroShotTM"):
        rng = np.random.default_rng(42)
        doc_emb = rng.standard_normal((len(_TOY), 8))
        model.fit(_TOY, doc_emb, iters=10)
        return model

    # Embedding models (FASTopic, BERTopic, Top2Vec): require doc_embeddings
    if name in ("FASTopic", "BERTopic", "Top2Vec"):
        rng = np.random.default_rng(42)
        doc_emb = rng.standard_normal((len(_TOY), 8))
        try:
            model.fit(_TOY, doc_emb, iters=10)
        except TypeError:
            # BERTopic/Top2Vec do not accept iters
            model.fit(_TOY, doc_emb)
        return model

    # LSA: a direct SVD solve; fit() takes no iters argument.
    if name in _DIRECT_SOLVE_MODELS:
        model.fit(_TOY)
        return model

    # All others: plain fit
    model.fit(_TOY, iters=10)
    return model


# ---------------------------------------------------------------------------
# Build parametrize list: (name, factory, family)
# ---------------------------------------------------------------------------

_PARAMS = []
for _name, _factory, _family in REGISTRY:
    _PARAMS.append((_name, _factory, _family))


def _ids():
    return [p[0] for p in _PARAMS]


# ---------------------------------------------------------------------------
# Test: fit_history attribute exists on class (sanity, covered by conformance)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_fit_history_class_attribute(name, factory, family):
    """fit_history must be a class-level attribute before fitting."""
    model = factory()
    assert hasattr(type(model), "fit_history"), (
        f"{name}: missing class attribute 'fit_history'"
    )


@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_converged_class_attribute(name, factory, family):
    """converged must be a class-level attribute before fitting."""
    model = factory()
    assert hasattr(type(model), "converged"), (
        f"{name}: missing class attribute 'converged'"
    )


# ---------------------------------------------------------------------------
# Test: fit_history type and structure after fitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_fit_history_is_list(name, factory, family):
    """fit_history must return a list after fitting."""
    model = _fit_model(name, factory)
    h = model.fit_history
    assert isinstance(h, list), f"{name}.fit_history is not a list: {type(h)}"


@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_fit_history_element_types(name, factory, family):
    """Each element of fit_history must be a (int, float) 2-tuple."""
    model = _fit_model(name, factory)
    h = model.fit_history
    for i, entry in enumerate(h):
        assert (
            isinstance(entry, (tuple, list)) and len(entry) == 2
        ), f"{name}.fit_history[{i}] is not a 2-tuple: {entry!r}"
        it, obj = entry
        assert isinstance(it, int) and it > 0, (
            f"{name}.fit_history[{i}] iteration is not a positive int: {it!r}"
        )
        assert isinstance(obj, float), (
            f"{name}.fit_history[{i}] objective is not a float: {obj!r}"
        )


# ---------------------------------------------------------------------------
# Test: fit_history non-empty for models that should record a trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_fit_history_non_empty(name, factory, family):
    """fit_history must be non-empty for models with a wired per-iteration trace.

    Part B models (parametric-Gibbs without a trace) are xfail.
    Cluster models (BERTopic, Top2Vec) and DTM are skipped (no trace by design).
    """
    # Cluster models and direct SVD solves: [] by design, not a gap
    if name in _NO_TRACE_MODELS:
        pytest.skip(f"{name}: no iterative objective; fit_history == [] by design")

    # DTM: no static per-iteration trace (time-sliced)
    if name == "DTM":
        pytest.skip("DTM: no static per-iteration log-likelihood trace")

    # HLDA: no flat K-topic trace
    if name == "HLDA":
        pytest.skip("HLDA: no flat K-topic objective trace")

    model = _fit_model(name, factory)
    h = model.fit_history
    assert len(h) > 0, (
        f"{name}.fit_history is empty after fitting; expected non-empty trace"
    )


# ---------------------------------------------------------------------------
# Test: converged type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_converged_type(name, factory, family):
    """converged must return bool for iterative models or None for cluster models."""
    model = _fit_model(name, factory)
    c = model.converged
    if name in _NO_TRACE_MODELS:
        assert c is None, f"{name}.converged should be None, got {c!r}"
    else:
        assert isinstance(c, bool), (
            f"{name}.converged should be bool, got {type(c).__name__}: {c!r}"
        )


# ---------------------------------------------------------------------------
# Test: converged semantics — default fit should not converge early
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=_ids())
def test_converged_false_by_default(name, factory, family):
    """Without convergence_tol, converged should be False (not True) after fitting.

    Cluster models are skipped (converged is None by design).
    """
    if name in _NO_TRACE_MODELS:
        pytest.skip(f"{name}: non-iterative model, converged is None")

    model = _fit_model(name, factory)
    c = model.converged
    assert c is False, (
        f"{name}.converged should be False after default fit (no convergence_tol), "
        f"got {c!r}"
    )


# ---------------------------------------------------------------------------
# LDA-specific: reference implementation tests
# ---------------------------------------------------------------------------

class TestLDAConvergenceInterface:
    """Detailed tests for the LDA reference convergence implementation."""

    def _fit(self, iters=100, convergence_tol=0.0, check_every=10, seed=42):
        model = topica.LDA(2, seed=seed)
        model.fit(
            _TOY,
            iters=iters,
            convergence_tol=convergence_tol,
            check_every=check_every,
        )
        return model

    def test_fit_history_non_empty_default(self):
        """Default fit (check_every=10) records a non-empty trace."""
        model = self._fit(iters=100)
        h = model.fit_history
        assert len(h) > 0

    def test_fit_history_iterations_are_multiples_of_check_every(self):
        """All recorded iterations are multiples of check_every."""
        check_every = 5
        model = self._fit(iters=50, check_every=check_every)
        h = model.fit_history
        assert len(h) > 0
        for it, _ in h:
            assert it % check_every == 0, (
                f"iteration {it} is not a multiple of check_every={check_every}"
            )

    def test_fit_history_len_equals_iters_div_check_every(self):
        """Trace has exactly iters // check_every entries (default, no early stop)."""
        iters, check_every = 100, 10
        model = self._fit(iters=iters, check_every=check_every)
        h = model.fit_history
        assert len(h) == iters // check_every

    def test_fit_history_objectives_are_finite(self):
        """All log-likelihood values in the trace must be finite."""
        import math
        model = self._fit(iters=100)
        for it, obj in model.fit_history:
            assert math.isfinite(obj), f"non-finite objective at iter {it}: {obj}"

    def test_fit_history_objectives_are_negative(self):
        """LDA log-likelihood is always negative."""
        model = self._fit(iters=100)
        for it, obj in model.fit_history:
            assert obj < 0, f"non-negative log-likelihood at iter {it}: {obj}"

    def test_fit_history_monotone_direction(self):
        """Log-likelihood should be non-decreasing overall (first vs last)."""
        model = self._fit(iters=200)
        h = model.fit_history
        assert h[-1][1] >= h[0][1], (
            f"Final LL ({h[-1][1]:.2f}) < initial LL ({h[0][1]:.2f}); "
            "expected improvement over training"
        )

    def test_converged_false_default(self):
        """Default fit (no convergence_tol) never sets converged=True."""
        model = self._fit(iters=100)
        assert model.converged is False

    def test_converged_true_with_tight_tol(self):
        """A very loose tolerance should trigger early convergence."""
        model = self._fit(iters=1000, convergence_tol=1.0, check_every=5)
        # With tol=1.0 (100% relative change is 'converged') the model should
        # stop well before 1000 iterations.
        assert model.converged is True

    def test_early_stop_reduces_trace_length(self):
        """When early-stop fires, trace is shorter than the full iters count."""
        iters, check_every = 1000, 5
        model = self._fit(iters=iters, convergence_tol=1.0, check_every=check_every)
        max_possible = iters // check_every
        assert len(model.fit_history) < max_possible, (
            "Expected trace to be shorter than max after early-stop"
        )

    def test_default_behavior_bit_for_bit(self):
        """Adding check_every/convergence_tol kwargs must not alter the default fit.

        Two models: one fit with no convergence kwargs (the historical default),
        one fit with explicit defaults (convergence_tol=0.0, check_every=10).
        They must produce identical topic_word matrices.
        """
        import numpy as np

        m1 = topica.LDA(2, seed=7)
        m1.fit(_TOY, iters=50)

        m2 = topica.LDA(2, seed=7)
        m2.fit(_TOY, iters=50, convergence_tol=0.0, check_every=10)

        np.testing.assert_array_equal(
            m1.topic_word, m2.topic_word,
            err_msg="topic_word differs: convergence kwargs altered the default path"
        )

    def test_check_every_zero_disables_trace(self):
        """check_every=0 disables trace recording entirely."""
        model = topica.LDA(2, seed=42)
        model.fit(_TOY, iters=50, check_every=0)
        assert model.fit_history == [], (
            "Expected empty fit_history when check_every=0"
        )

    def test_fit_history_persists_across_save_load(self, tmp_path):
        """fit_history must survive a save/load round-trip."""
        model = self._fit(iters=50)
        path = str(tmp_path / "lda_convergence.bin")
        model.save(path)
        loaded = topica.LDA.load(path)
        assert loaded.fit_history == model.fit_history, (
            "fit_history does not match after save/load"
        )

    def test_converged_persists_across_save_load(self, tmp_path):
        """converged must survive a save/load round-trip."""
        model = self._fit(iters=1000, convergence_tol=1.0, check_every=5)
        assert model.converged is True  # precondition
        path = str(tmp_path / "lda_converged.bin")
        model.save(path)
        loaded = topica.LDA.load(path)
        assert loaded.converged is True


# ---------------------------------------------------------------------------
# keyATM opt-in convergence (issue #95): base, covariate, and dynamic Gibbs
# variants early-stop on convergence_tol; CVB0 never does.
# ---------------------------------------------------------------------------

class TestKeyATMConvergence:
    """keyATM mirrors the LDA-family convergence contract on its Gibbs backends."""

    def _base(self, iters=400, convergence_tol=0.0, seed=42):
        m = topica.KeyATM({"a": ["cat"]}, num_topics=2, seed=seed)
        m.fit(_TOY, iters=iters, convergence_tol=convergence_tol)
        return m

    def test_default_converged_false_and_trace_nonempty(self):
        """Default fit (no tol) runs to the cap: converged False, trace recorded."""
        m = self._base(iters=400)
        assert m.converged is False
        assert len(m.fit_history) > 0

    def test_tol_triggers_early_stop(self):
        """A loose tolerance stops the base sampler well before the cap."""
        m = self._base(iters=400, convergence_tol=1.0)
        assert m.converged is True
        assert m.fit_history[-1][0] < 400

    def test_covariate_variant_early_stops(self):
        """The covariate (DMR-prior) keyATM honors convergence_tol."""
        import numpy as np
        m = topica.KeyATM({"a": ["cat"]}, num_topics=2, seed=42)
        cov = np.array([[1.0, 0.0]] * 15 + [[0.0, 1.0]] * 15)
        m.fit(_TOY, covariates=cov, iters=400, convergence_tol=1.0)
        assert m.converged is True

    def test_dynamic_variant_early_stops(self):
        """The dynamic (change-point HMM) keyATM honors convergence_tol."""
        m = topica.KeyATM({"a": ["cat"]}, num_topics=2, seed=42)
        ts = [0] * 15 + [1] * 15
        m.fit(_TOY, timestamps=ts, num_states=2, iters=400, convergence_tol=1.0)
        assert m.converged is True

    def test_cvb0_ignores_tol(self):
        """The CVB0 backend keeps no trace and never early-stops."""
        m = topica.KeyATM({"a": ["cat"]}, num_topics=2, seed=42, sampler="cvb0")
        m.fit(_TOY, iters=50, convergence_tol=1.0)
        assert m.converged is False

    def test_default_behavior_bit_for_bit(self):
        """Passing convergence_tol=0.0 must not change the default fit."""
        import numpy as np
        a = self._base(iters=200, convergence_tol=0.0, seed=7)
        b = topica.KeyATM({"a": ["cat"]}, num_topics=2, seed=7)
        b.fit(_TOY, iters=200)
        np.testing.assert_array_equal(a.topic_word, b.topic_word)

    def test_converged_persists_across_save_load(self, tmp_path):
        m = self._base(iters=400, convergence_tol=1.0)
        assert m.converged is True  # precondition
        path = str(tmp_path / "keyatm_converged.bin")
        m.save(path)
        loaded = topica.KeyATM.load(path)
        assert loaded.converged is True


# ---------------------------------------------------------------------------
# Cluster models: explicit type checks
# ---------------------------------------------------------------------------

class TestClusterModelConvergence:
    """BERTopic and Top2Vec must return [] and None."""

    @staticmethod
    def _doc_embeddings():
        import numpy as np
        return np.random.default_rng(42).standard_normal((len(_TOY), 8))

    def test_bertopic_fit_history_empty(self):
        model = topica.BERTopic(min_cluster_size=5)
        model.fit(_TOY, self._doc_embeddings())
        assert model.fit_history == []

    def test_top2vec_fit_history_empty(self):
        model = topica.Top2Vec()
        model.fit(_TOY, self._doc_embeddings())
        assert model.fit_history == []

    def test_bertopic_converged_none(self):
        model = topica.BERTopic(min_cluster_size=5)
        model.fit(_TOY, self._doc_embeddings())
        assert model.converged is None

    def test_top2vec_converged_none(self):
        model = topica.Top2Vec()
        model.fit(_TOY, self._doc_embeddings())
        assert model.converged is None
