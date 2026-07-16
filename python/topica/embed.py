"""Document-embedding helpers for the embedding-based models.

Seed, compute, save, and load the document/word embeddings that ETM,
EmbeddingLDA, BERTopic, Top2Vec, and the other embedding models consume.
"""

from __future__ import annotations

from .embedding import (
    embedding_seeds,
    llm_embed,
    save_embeddings,
    load_embeddings,
)

__all__ = ["embedding_seeds", "llm_embed", "save_embeddings", "load_embeddings"]
