"""The topica model roster, namespaced.

Every estimator in the library, reachable as ``topica.models.<Name>`` -- the
organized home for the model classes as the top-level namespace is decluttered.
``topica.models.LDA`` and ``from topica.models import LDA, STM`` both work.

The roster is the same set the registry describes; :func:`topica.list_models`
documents each one's group, inference engine, and requirements.
"""

from __future__ import annotations

# Compiled estimators (Rust core, via the PyO3 extension).
from ._topica import (
    LDA,
    CTM,
    ProdLDA,
    HDP,
    NMF,
    LSA,
    TensorLDA,
    STM,
    STS,
    SAGE,
    ECTM,
    DMR,
    KeyATM,
    SeededLDA,
    LabeledLDA,
    SupervisedLDA,
    GSDMM,
    PT,
    DTM,
    DETM,
    HLDA,
    PA,
    BERTopic,
    Top2Vec,
    ETM,
    IdealPointTM,
    Wordfish,
    IdealPointSentenceTM,
    TBIP,
    PartyEmbeddings,
    FASTopic,
    CombinedTM,
    ZeroShotTM,
    InfoCTM,
)

# Pure-Python wrappers (thin layers over the core or external backends).
from .anchor import AnchorLDA
from .gdmr import GDMR
from .narrative import NarrativeTM
from .embedding import EmbeddingLDA
from .topicgpt import TopicGPT

__all__ = [
    "LDA",
    "CTM",
    "ProdLDA",
    "HDP",
    "NMF",
    "LSA",
    "TensorLDA",
    "STM",
    "STS",
    "SAGE",
    "ECTM",
    "DMR",
    "KeyATM",
    "SeededLDA",
    "LabeledLDA",
    "SupervisedLDA",
    "GSDMM",
    "PT",
    "DTM",
    "DETM",
    "HLDA",
    "PA",
    "BERTopic",
    "Top2Vec",
    "ETM",
    "IdealPointTM",
    "Wordfish",
    "IdealPointSentenceTM",
    "TBIP",
    "PartyEmbeddings",
    "FASTopic",
    "CombinedTM",
    "ZeroShotTM",
    "InfoCTM",
    "AnchorLDA",
    "GDMR",
    "NarrativeTM",
    "EmbeddingLDA",
    "TopicGPT",
]
