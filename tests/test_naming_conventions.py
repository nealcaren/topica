"""Naming-convention lint: the shared vocabulary, enforced.

A hand-maintained style guide drifts. This test is the source of truth for
topica's cross-model API vocabulary; ``docs/contributing/conventions.md`` is the
explanation. It introspects every model's ``__init__`` and ``fit`` signature and
fails when a new model breaks a convention, so conceptual integrity is checked by
CI rather than by memory.

Principled exceptions and tracked drift are recorded below (mirroring
``conformance.py``): EXCEPT sets are structural and permanent; KNOWN_DRIFT is the
burn-down worklist of things we intend to align later, kept here so the suite is
green today while still documenting what is not yet conformant.
"""

from __future__ import annotations

import inspect

import pytest

import topica


def _model_classes():
    out = []
    for name in dir(topica):
        obj = getattr(topica, name)
        if isinstance(obj, type) and hasattr(obj, "fit") and hasattr(obj, "num_topics"):
            out.append((name, obj))
    return sorted(out)


MODELS = _model_classes()
MODEL_IDS = [n for n, _ in MODELS]


def _ctor_params(cls):
    return list(inspect.signature(cls).parameters.values())


def _fit_params(cls):
    return [p for p in inspect.signature(cls.fit).parameters.values() if p.name != "self"]


def _all_param_names(cls):
    names = {p.name for p in _ctor_params(cls)}
    names |= {p.name for p in _fit_params(cls)}
    return names


# --- Forbidden synonyms: a concept must use the canonical name (value), never
#     the key. This is the strongest guard against new drift. ---
FORBIDDEN = {
    "iterations": "iters",
    "n_iter": "iters",
    "max_iter": "iters",
    "epochs": "iters",
    "random_state": "seed",
    "random_seed": "seed",
    "n_topics": "num_topics",
    "ntopics": "num_topics",
    "tol": "convergence_tol",
}

# Constructors whose first positional is not ``num_topics``. Each is principled:
# K is discovered, or a different required input leads.
CTOR_FIRST_EXCEPT = {
    "BERTopic": "K discovered from clustering",
    "HDP": "K discovered (nonparametric)",
    "Top2Vec": "K discovered from clustering",
    "TopicGPT": "K discovered by the LLM; keyword-only constructor (backend/model)",
    "LabeledLDA": "topics are the label set, not a K argument",
    "KeyATM": "keyword dict is the leading required input",
    "SeededLDA": "seed-word dict is the leading required input",
    "PA": "two-level model: num_super, num_sub",
    "MGLDA": "two-grain model: num_global_topics, num_local_topics",
    "DiscLDA": "structured topic count: num_classes*k_class + k_shared, "
    "num_classes known only at fit",
    "FactorialLDA": "factorial model: factor_sizes is the leading required input; "
    "num_topics is the derived tuple count prod(factor_sizes)",
    "FLDA": "alias of FactorialLDA (same class); factor_sizes is the leading input",
}

# Models that take a document covariate design matrix. Each must accept a
# ``covariates=`` keyword (the canonical cross-model alias), regardless of its
# native primary name (features / prevalence) kept for reference-package fidelity.
COVARIATE_MODELS = {"DMR", "GDMR", "STM", "STS", "KeyATM"}

# Models that take a per-document time index. Each must accept a ``times=``
# keyword (the canonical cross-model name). KeyATM keeps ``timestamps=`` as an
# accepted alias.
TEMPORAL_MODELS = {"DTM", "DETM", "KeyATM", "TopicsOverTime"}

# Tracked drift: (model, param) -> reason. These are known and intentionally not
# yet aligned; remove an entry once the drift is fixed. The test treats them as
# allowed so the suite stays green and this map is the worklist. Empty == the
# burn-down is clear.
KNOWN_DRIFT: dict = {}


@pytest.mark.parametrize("name,cls", MODELS, ids=MODEL_IDS)
def test_no_forbidden_synonyms(name, cls):
    """No model uses a non-canonical synonym for a shared concept."""
    bad = []
    for p in _ctor_params(cls):
        if p.name in FORBIDDEN and (name, p.name) not in KNOWN_DRIFT:
            bad.append((p.name, FORBIDDEN[p.name]))
    for p in _fit_params(cls):
        if p.name in FORBIDDEN and (name, p.name) not in KNOWN_DRIFT:
            bad.append((p.name, FORBIDDEN[p.name]))
    assert not bad, (
        f"{name} uses non-canonical parameter name(s): "
        + ", ".join(f"{got!r} (use {want!r})" for got, want in bad)
    )


@pytest.mark.parametrize("name,cls", MODELS, ids=MODEL_IDS)
def test_seed_is_named_seed_and_defaults_to_13(name, cls):
    """The RNG seed is always ``seed`` and always defaults to 13."""
    params = {p.name: p for p in _ctor_params(cls)}
    if "seed" not in params:
        return  # a model with no seed is its own (rare) case
    assert params["seed"].default == 13, (
        f"{name}: seed default is {params['seed'].default!r}, expected 13"
    )


@pytest.mark.parametrize("name,cls", MODELS, ids=MODEL_IDS)
def test_num_topics_is_first_positional(name, cls):
    """The constructor's first positional argument is ``num_topics``."""
    if name in CTOR_FIRST_EXCEPT:
        return
    positional = [
        p.name for p in _ctor_params(cls)
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
    ]
    assert positional and positional[0] == "num_topics", (
        f"{name}: first positional is {positional[:1]}, expected ['num_topics'] "
        f"(or add to CTOR_FIRST_EXCEPT with a reason)"
    )


@pytest.mark.parametrize("name", sorted(COVARIATE_MODELS))
def test_covariate_models_accept_covariates_alias(name):
    """Every covariate model accepts ``covariates=`` on fit, the canonical
    cross-model alias, whatever its native primary name is."""
    cls = getattr(topica, name)
    fit_names = {p.name for p in _fit_params(cls)}
    assert "covariates" in fit_names, (
        f"{name}.fit must accept a 'covariates=' alias for the document "
        f"covariate design matrix"
    )


@pytest.mark.parametrize("name", sorted(TEMPORAL_MODELS))
def test_temporal_models_accept_times(name):
    """Every temporal model accepts a ``times`` argument (the canonical name);
    ``timestamps`` may remain as an alias but ``times`` must exist."""
    cls = getattr(topica, name)
    fit_names = {p.name for p in _fit_params(cls)}
    assert "times" in fit_names, (
        f"{name}.fit must accept 'times' (the canonical per-document time index)"
    )


def test_known_drift_entries_are_real():
    """Guard the worklist: every KNOWN_DRIFT entry must still exist, so the map
    is cleaned up as drift is fixed rather than rotting."""
    for (mname, pname) in KNOWN_DRIFT:
        cls = getattr(topica, mname, None)
        assert cls is not None, f"KNOWN_DRIFT names unknown model {mname!r}"
        present = pname in _all_param_names(cls)
        assert present, (
            f"KNOWN_DRIFT[{(mname, pname)}] no longer applies "
            f"({mname} has no '{pname}'); remove the entry"
        )
