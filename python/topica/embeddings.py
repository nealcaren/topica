"""Embedding infrastructure and embedding-based analysis (workflow namespace, #757).

Embedding I/O, embedding-guided topic models, and conText-style embedding
regression. Re-exports helpers from :mod:`topica.embedding`,
:mod:`topica.embedding_regression`, and :mod:`topica.anchor`; the same names are
available at the package root.
"""

from __future__ import annotations

from .embedding import (
    EmbeddingLDA, embedding_seeds, llm_embed, save_embeddings, load_embeddings,
)
from .embedding_regression import (
    EmbeddingRegression, embedding_regression, alc_embeddings, compute_transform,
)
from .anchor import AnchorLDA

__all__ = [
    "EmbeddingLDA",
    "embedding_seeds",
    "llm_embed",
    "save_embeddings",
    "load_embeddings",
    "EmbeddingRegression",
    "embedding_regression",
    "alc_embeddings",
    "compute_transform",
    "AnchorLDA",
]
