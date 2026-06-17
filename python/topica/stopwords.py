"""Bundled stopword lists.

``ENGLISH_STOPWORDS`` is a small frozenset of common English function words that
rarely carry topical meaning. Pass it to :func:`topica.tokenize` (or the corpus
builders) so a first LDA fit is not dominated by ``the`` / ``and`` / ``of``::

    docs = [topica.tokenize(t, stopwords=topica.ENGLISH_STOPWORDS) for t in texts]

For other languages — or a more comprehensive English list — use
:func:`stopwords`, which serves the `stopwords-iso
<https://github.com/stopwords-iso/stopwords-iso>`_ lists (58 languages, MIT
licensed; see ``_data/STOPWORDS_ISO_LICENSE.txt``)::

    fr = topica.stopwords("fr")                 # or "french"
    corpus = topica.from_dataframe(df, text_col="texte", stopwords=fr)
    topica.stopword_languages()                 # the available ISO 639-1 codes

``ENGLISH_STOPWORDS`` is kept as the short, stable default; ``stopwords("en")``
is the larger stopwords-iso English list.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).parent / "_data" / "stopwords_iso.json"

# ISO 639-1 code aliases for common English language names, so social scientists
# can pass "french" as well as "fr".
_NAME_TO_CODE = {
    "arabic": "ar", "bengali": "bn", "bulgarian": "bg", "catalan": "ca",
    "chinese": "zh", "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "esperanto": "eo", "estonian": "et",
    "finnish": "fi", "french": "fr", "galician": "gl", "german": "de",
    "greek": "el", "gujarati": "gu", "hebrew": "he", "hindi": "hi",
    "hungarian": "hu", "indonesian": "id", "italian": "it", "japanese": "ja",
    "korean": "ko", "latin": "la", "latvian": "lv", "lithuanian": "lt",
    "malay": "ms", "marathi": "mr", "norwegian": "no", "persian": "fa",
    "polish": "pl", "portuguese": "pt", "romanian": "ro", "russian": "ru",
    "slovak": "sk", "slovenian": "sl", "somali": "so", "spanish": "es",
    "swahili": "sw", "swedish": "sv", "tagalog": "tl", "thai": "th",
    "turkish": "tr", "ukrainian": "uk", "urdu": "ur", "vietnamese": "vi",
}


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_DATA, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def stopwords(language: str) -> frozenset:
    """Return the bundled stopword list for ``language`` as a ``frozenset``.

    ``language`` is an ISO 639-1 code (``"fr"``, ``"de"``, ``"zh"``) or a common
    English name (``"french"``, ``"german"``, ``"chinese"``), case-insensitive.
    The lists come from `stopwords-iso
    <https://github.com/stopwords-iso/stopwords-iso>`_ (MIT licensed). Pass the
    result to :func:`topica.tokenize` or the corpus builders::

        corpus = topica.from_dataframe(df, text_col="texte",
                                       stopwords=topica.stopwords("fr"))

    Raises ``ValueError`` with the available codes if the language is unknown.
    """
    key = language.strip().lower()
    code = _NAME_TO_CODE.get(key, key)
    data = _load()
    if code not in data:
        raise ValueError(
            f"no bundled stopword list for {language!r}. Available ISO 639-1 "
            f"codes: {', '.join(sorted(data))}. (Pass your own list for anything "
            f"not covered.)"
        )
    return frozenset(data[code])


def stopword_languages() -> list:
    """Sorted list of ISO 639-1 codes with a bundled :func:`stopwords` list."""
    return sorted(_load())


ENGLISH_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also",
    "although", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "even", "few",
    "for", "from", "further", "get", "had", "has", "have", "having", "he",
    "her", "here", "him", "his", "how", "however", "if", "in", "into", "is",
    "it", "its", "itself", "just", "many", "more", "most", "much", "must",
    "no", "not", "now", "of", "on", "once", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "should", "since", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "you", "your",
})
