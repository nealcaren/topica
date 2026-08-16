"""Corpus construction and preparation (workflow namespace, issue #757).

Everything for getting text into a :class:`~topica.Corpus`: DataFrame ingest with
aligned metadata, tokenization, phrase/collocation learning, stopword lists, and
the bundled example datasets. Re-exports helpers that live in
:mod:`topica.frames`, :mod:`topica.preprocess`, :mod:`topica.phrases`, and
:mod:`topica.stopwords`; the same names are available at the package root.
"""

from __future__ import annotations

from ._topica import Corpus, tokenize
from .frames import from_dataframe, align, prep_documents, plot_removed
from .preprocess import split_documents
from .phrases import learn_phrases, apply_phrases, add_ngrams, export_phrases, Phrases
from .stopwords import (
    ENGLISH_STOPWORDS, SENTIMENT_STOPWORDS, stopwords, stopword_languages,
)
from . import datasets

__all__ = [
    "Corpus",
    "tokenize",
    "from_dataframe",
    "align",
    "prep_documents",
    "plot_removed",
    "split_documents",
    "learn_phrases",
    "apply_phrases",
    "add_ngrams",
    "export_phrases",
    "Phrases",
    "ENGLISH_STOPWORDS",
    "SENTIMENT_STOPWORDS",
    "stopwords",
    "stopword_languages",
    "datasets",
]
