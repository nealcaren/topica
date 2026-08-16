"""Analysis provenance and model discovery (workflow namespace, issue #757).

Recording what was fit and how (the analysis manifest), the estimator conformance
contract, and the model registry. Re-exports helpers from :mod:`topica.manifest`,
:mod:`topica.conformance`, and :mod:`topica.registry`; the same names are
available at the package root.
"""

from __future__ import annotations

from .manifest import AnalysisManifest, record_fit
from .conformance import check_conformance
from .registry import list_models, ModelInfo, REGISTRY, effective_determinism

__all__ = [
    "AnalysisManifest",
    "record_fit",
    "check_conformance",
    "list_models",
    "ModelInfo",
    "REGISTRY",
    "effective_determinism",
]
