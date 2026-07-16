"""Corpus construction and text preparation.

Turn raw text or a dataframe into the token lists and :class:`~topica.Corpus`
the models fit on: split, align, learn multi-word phrases, drop stopwords, and
restrict a new corpus to a fitted model's vocabulary before ``transform``.
"""

from __future__ import annotations

from .preprocess import split_documents
from .frames import from_dataframe, align, prep_documents
from .phrases import learn_phrases, apply_phrases, add_ngrams, Phrases
from .stopwords import ENGLISH_STOPWORDS, stopwords, stopword_languages
from .stm import align_corpus

__all__ = [
    "split_documents",
    "from_dataframe",
    "align",
    "prep_documents",
    "learn_phrases",
    "apply_phrases",
    "add_ngrams",
    "Phrases",
    "align_corpus",
    "stopwords",
    "stopword_languages",
    "ENGLISH_STOPWORDS",
]
