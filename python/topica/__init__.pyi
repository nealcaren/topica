from __future__ import annotations

from ._topica import (
    Corpus as Corpus,
    tokenize as tokenize,
    project as project,
    DEFAULT_TOKEN_REGEX as DEFAULT_TOKEN_REGEX,
    __version__ as __version__,
)
from .registry import (
    list_models as list_models,
    ModelInfo as ModelInfo,
    REGISTRY as REGISTRY,
)

# Role namespaces -- the organized public API. Each is a module; the callable
# surface lives inside (e.g. topica.diagnostics.coherence, topica.models.LDA).
from . import models as models
from . import prep as prep
from . import design as design
from . import embed as embed
from . import diagnostics as diagnostics
from . import content as content
from . import select as select
from . import interpret as interpret
from . import effects as effects
from . import scaling as scaling
from . import ensemble as ensemble
from . import llm as llm
from . import viz as viz
# Model-specific helper toolkits.
from . import stm as stm
from . import keyatm as keyatm
from . import ectm as ectm
from . import datasets as datasets

__citation__: str


def enable_experimental(enabled: bool = True) -> None: ...
def experimental_enabled() -> bool: ...
