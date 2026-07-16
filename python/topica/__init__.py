"""topica: fast SparseLDA topic modeling (MALLET's algorithm) in Rust.

The heavy lifting lives in the compiled extension ``topica._topica``;
this module just re-exports its public surface so ``import topica`` works
and editors/type-checkers see a stable namespace.
"""

# The model classes live under ``topica.models`` (topica.models.LDA); the flat
# top level carries only the corpus, tokenizer, and process-wide helpers.
from ._topica import (
    Corpus,
    tokenize,
    project,
    set_experimental as _set_experimental,
    experimental_is_enabled as _experimental_is_enabled,
    DEFAULT_TOKEN_REGEX,
    __version__,
)


def enable_experimental(enabled: bool = True) -> None:
    """Opt into experimental, unvalidated models for this process.

    Some models ship before they have a published paper and a
    reference-implementation parity check (topica's bar for a *validated*
    model). They are flagged **experimental**: kept out of the validated roster
    and the README model table, documented separately, and refused at
    construction or load until you opt in here. Call this once, early, before
    constructing an experimental model; pass ``False`` to turn the gate back
    on. Use :func:`list_models` with ``experimental=True`` to see the current
    set. Equivalent to setting the ``TOPICA_EXPERIMENTAL=1``
    environment variable. Experimental models may change or be removed without a
    deprecation cycle.
    """
    _set_experimental(bool(enabled))


def experimental_enabled() -> bool:
    """Whether experimental models are currently enabled (see
    :func:`enable_experimental`)."""
    return bool(_experimental_is_enabled())

__citation__ = (
    "Caren, N. (2026). topica: fast, all-purpose topic modeling for Python. "
    "https://github.com/nealcaren/topica\n\n"
    "@software{caren_topica,\n"
    "  author = {Caren, Neal},\n"
    "  title  = {topica: fast, all-purpose topic modeling for Python},\n"
    "  year   = {2026},\n"
    "  url    = {https://github.com/nealcaren/topica}\n"
    "}\n\n"
    "Please also cite the model(s) you use; see "
    "https://nealcaren.github.io/topica/citing/."
)


# --- The public API is organized by role: what you are doing, not which model.
# Each name below is a namespace; e.g. topica.diagnostics.coherence,
# topica.effects.estimate_effect, topica.models.LDA. ---
from . import models  # noqa: E402       the estimator roster (topica.models.LDA)
from . import prep  # noqa: E402         corpus construction & text prep
from . import design  # noqa: E402       covariate design matrices
from . import embed  # noqa: E402        document-embedding helpers
from . import diagnostics  # noqa: E402  quality / stability / held-out / MCMC convergence
from . import content  # noqa: E402      content-covariate diagnostics (STM/STS/SAGE/ECTM)
from . import select  # noqa: E402       choosing K / choosing a model
from . import interpret  # noqa: E402    reading & labeling topics
from . import effects  # noqa: E402      covariate / prevalence estimation
from . import scaling  # noqa: E402      ideal-point scaling diagnostics
from . import ensemble  # noqa: E402     combining runs
from . import llm  # noqa: E402          LLM-based evaluation (topica.llm.*)
# viz (all plotting) is heavier; it is imported lazily on first access to
# topica.viz (see __getattr__ below) so `import topica` stays light.

# Model-specific helper toolkits, kept as their own namespaces (not roles):
from . import stm  # noqa: E402          STM/CTM covariate-design + effect helpers
from . import keyatm  # noqa: E402       keyATM-specific workflow helpers
from . import ectm  # noqa: E402         ECTM content-trajectory helpers

# Discovery + bundled data.
from .registry import list_models, ModelInfo, REGISTRY  # noqa: E402
from . import datasets  # noqa: E402


def __getattr__(name):
    # Lazily expose topica.viz so importing matplotlib is deferred until first use.
    # import_module (not `from . import viz`) avoids re-entering this __getattr__.
    if name == "viz":
        import importlib
        return importlib.import_module(".viz", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # core entry points
    "Corpus",
    "tokenize",
    "project",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
    "__citation__",
    # discovery
    "list_models",
    "ModelInfo",
    "REGISTRY",
    "enable_experimental",
    "experimental_enabled",
    # role namespaces (the organized public API)
    "models",
    "prep",
    "design",
    "embed",
    "diagnostics",
    "content",
    "select",
    "interpret",
    "effects",
    "scaling",
    "ensemble",
    "llm",
    "viz",
    # model-specific helper toolkits
    "stm",
    "keyatm",
    "ectm",
    # bundled data
    "datasets",
]


def __dir__():
    # Present only the curated public surface to dir()/tab-completion. The
    # internal leaf modules (coherence, validation, analysis, ...) remain
    # importable but are hidden to keep the namespace clean.
    return sorted(__all__)
