"""#400 — uniform ``model.settings`` introspection.

Every public model exposes its constructor configuration as a JSON-serialisable
dict, keyword-named to match ``__init__``. The expected key set is derived
directly from the class's ``__text_signature__`` (minus the data-guidance args),
so this test is the single inventory: adding a constructor parameter without
surfacing it in ``settings`` fails here, with no second list to maintain.
"""

from __future__ import annotations

import inspect
import json

import pytest

import topica
import topica._topica as _ext
from topica.registry import REGISTRY

# Several registered models are experimental-gated and refuse to construct
# otherwise; settings introspection is part of their surface too.
topica.enable_experimental()

# Constructor parameters that are data / guidance inputs, not hyperparameters,
# and are therefore intentionally omitted from ``settings`` (recorded elsewhere
# as corpus/guidance data).
_DATA_ARGS: dict[str, set[str]] = {
    "KeyATM": {"keywords"},
    "SeededLDA": {"seed_words"},
    "Scholar": {"covariates", "content"},
    "EmbeddingLDA": {"embeddings", "vocabulary"},
    "TopicGPT": {"backend"},
}

# Zero/low-arg factories for every Rust-backed public model. Guidance/label args
# get minimal placeholders; the values do not matter — settings reports the
# *construction config*, which is available before fit.
_FACTORIES: dict[str, object] = {
    "LDA": lambda: topica.LDA(2),
    "OnlineLDA": lambda: topica.OnlineLDA(2),
    "DMR": lambda: topica.DMR(2),
    "SAGE": lambda: topica.SAGE(2),
    "PA": lambda: topica.PA(num_super=2, num_sub=4),
    "PT": lambda: topica.PT(num_topics=2, num_pseudo=10),
    "HDP": lambda: topica.HDP(),
    "HLDA": lambda: topica.HLDA(),
    "LabeledLDA": lambda: topica.LabeledLDA(),
    "SupervisedLDA": lambda: topica.SupervisedLDA(num_topics=2),
    "DTM": lambda: topica.DTM(2),
    "DiscLDA": lambda: topica.DiscLDA(2, 2),
    "KeyATM": lambda: topica.KeyATM({"a": ["x"]}, num_topics=2),
    "SeededLDA": lambda: topica.SeededLDA({"a": ["x"], "b": ["y"]}),
    "GSDMM": lambda: topica.GSDMM(num_topics=5),
    "BTM": lambda: topica.BTM(2),
    "FactorialLDA": lambda: topica.FactorialLDA([3, 2]),
    "STM": lambda: topica.STM(2),
    "CTM": lambda: topica.CTM(2),
    "STS": lambda: topica.STS(2),
    "ETM": lambda: topica.ETM(2),
    "DETM": lambda: topica.DETM(2),
    "ProdLDA": lambda: topica.ProdLDA(2),
    "CombinedTM": lambda: topica.CombinedTM(2),
    "ZeroShotTM": lambda: topica.ZeroShotTM(2),
    "FASTopic": lambda: topica.FASTopic(2),
    "InfoCTM": lambda: topica.InfoCTM(2),
    "NMF": lambda: topica.NMF(2),
    "LSA": lambda: topica.LSA(2),
    "BERTopic": lambda: topica.BERTopic(min_cluster_size=5),
    "Top2Vec": lambda: topica.Top2Vec(),
    "SemanticSignalSeparation": lambda: topica.SemanticSignalSeparation(2),
    "PolylingualLDA": lambda: topica.PolylingualLDA(2),
    "Scholar": lambda: topica.Scholar(2),
    "RTM": lambda: topica.RTM(2),
    "TensorLDA": lambda: topica.TensorLDA(2),
    "TBIP": lambda: topica.TBIP(2),
    "IdealPointTM": lambda: topica.IdealPointTM(2),
    "IdealPointSentenceTM": lambda: topica.IdealPointSentenceTM(2),
    "Wordfish": lambda: topica.Wordfish(),
    "PartyEmbeddings": lambda: topica.PartyEmbeddings(),
    # Pure-Python wrapper models.
    "GDMR": lambda: topica.GDMR(2, degrees=[2]),
    "NarrativeTM": lambda: topica.NarrativeTM(2),
    "AnchorLDA": lambda: topica.AnchorLDA(2),
    "EmbeddingLDA": lambda: topica.EmbeddingLDA(
        2, embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], vocabulary=["a", "b", "c"]
    ),
    "TopicGPT": lambda: topica.TopicGPT(backend=lambda p: ""),
    "MechanisticLDA": lambda: topica.MechanisticLDA(2),
    "MechanisticBERTopic": lambda: topica.MechanisticBERTopic(min_cluster_size=5),
}

# Every public registry model (Rust aliases collapse to one qualname).
_MODELS = sorted(
    set({getattr(_ext, n).__qualname__: n for n in REGISTRY if hasattr(_ext, n)}.values())
    | {n for n in REGISTRY if not hasattr(_ext, n)}
)


def _expected_keys(cls) -> set[str]:
    """Public keyword/positional constructor parameter names.

    ``inspect.signature`` reads the Rust ``__text_signature__`` and native Python
    ``__init__`` alike, so this covers both binding kinds. VAR_KEYWORD/VAR_POSITIONAL
    (``*args``/``**kwargs``) are skipped.
    """
    out = set()
    for p in inspect.signature(cls).parameters.values():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        out.add(p.name)
    return out


def test_every_registry_model_covered():
    """No public registry model is missing a factory here."""
    missing = [n for n in _MODELS if n not in _FACTORIES]
    assert not missing, f"add factories for: {missing}"


@pytest.mark.parametrize("name", _MODELS)
def test_settings_present_and_json(name):
    model = _FACTORIES[name]()
    assert hasattr(model, "settings"), f"{name} has no .settings"
    s = model.settings
    assert isinstance(s, dict), f"{name}.settings is {type(s)}, not dict"
    # strict JSON: no NaN/Inf, no numpy scalars, no Python objects
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize("name", _MODELS)
def test_settings_keys_match_constructor(name):
    cls = getattr(topica, name)
    expected = _expected_keys(cls) - _DATA_ARGS.get(name, set())
    got = set(_FACTORIES[name]().settings.keys())
    assert got == expected, (
        f"{name}.settings keys mismatch\n"
        f"  missing: {sorted(expected - got)}\n"
        f"  extra:   {sorted(got - expected)}"
    )


# Non-default value cases for the reconstruction-prone params: enum-ish strings
# rebuilt from internal flags/enums, renamed fields, and deprecated aliases.
# ``settings`` must report the effective public value, not an internal form.
_VALUE_CASES = [
    (lambda: topica.LDA(2, sampler="warp"), "sampler", "warp"),
    (lambda: topica.LDA(2, sampler="lightlda"), "sampler", "lightlda"),
    (lambda: topica.LDA(2, sampler="cvb0"), "sampler", "cvb0"),
    (lambda: topica.LDA(2, num_threads=0), "num_threads", 1),  # .max(1) floor
    (lambda: topica.LDA(2, init="spectral"), "init", "spectral"),
    (lambda: topica.DMR(2, sampler="warp"), "sampler", "warp"),
    (lambda: topica.DMR(2, sampler="cvb0"), "sampler", "cvb0"),
    (lambda: topica.LabeledLDA(sampler="cvb0"), "sampler", "cvb0"),
    (lambda: topica.SeededLDA({"a": ["x"], "b": ["y"]}, sampler="warp"), "sampler", "warp"),
    (lambda: topica.KeyATM({"a": ["x"]}, num_topics=2, sampler="cvb0"), "sampler", "cvb0"),
    (lambda: topica.STM(2, init="random"), "init", "random"),
    (lambda: topica.STM(2, variational="diagonal"), "variational", "diagonal"),
    (lambda: topica.CTM(2, init="random"), "init", "random"),
    (lambda: topica.DTM(2, init="spectral"), "init", "spectral"),
    (lambda: topica.STS(2, init="random"), "init", "random"),
    (lambda: topica.NMF(2, beta_loss="kullback-leibler"), "beta_loss", "kullback-leibler"),
    (lambda: topica.NMF(2, weighting="tfidf"), "weighting", "tfidf"),
    (lambda: topica.NMF(2, init="random"), "init", "random"),
    (lambda: topica.LSA(2, weighting="count"), "weighting", "count"),
    (lambda: topica.ProdLDA(2, prior="dirichlet"), "prior", "dirichlet"),
    (lambda: topica.ETM(2, prior="laplace"), "prior", "laplace"),
    (lambda: topica.ETM(2, inference="vae"), "inference", "vae"),
    (lambda: topica.BERTopic(reducer="umap", min_cluster_size=5), "reducer", "umap"),
    (lambda: topica.BERTopic(clusterer="kmeans", num_clusters=3, min_cluster_size=5), "clusterer", "kmeans"),
    (lambda: topica.BERTopic(metric="euclidean", min_cluster_size=5), "metric", "euclidean"),
    (lambda: topica.HDP(beta=0.5), "beta", 0.5),          # stored in the field named `eta`
    (lambda: topica.HDP(beta=0.5), "eta", None),          # deprecated alias always None
    (lambda: topica.GSDMM(num_topics=7), "num_topics", 7),  # stored as `k_max`
    (lambda: topica.FASTopic(2, convergence_tol=1e-3), "convergence_tol", 1e-3),  # stored in `em_tol`
    (lambda: topica.FASTopic(2, convergence_tol=1e-3), "em_tol", None),  # deprecated alias always None
]


@pytest.mark.parametrize("factory,key,expected", _VALUE_CASES)
def test_settings_reconstructed_values(factory, key, expected):
    assert factory().settings[key] == expected
