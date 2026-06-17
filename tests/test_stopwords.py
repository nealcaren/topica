"""Tests for the bundled stopword lists (topica.stopwords / stopword_languages)."""

import pytest

import topica


def test_english_stopwords_unchanged():
    # The short curated default is stable (back-compat); it is NOT the iso list.
    assert isinstance(topica.ENGLISH_STOPWORDS, frozenset)
    assert "the" in topica.ENGLISH_STOPWORDS
    assert len(topica.ENGLISH_STOPWORDS) == 115


def test_languages_listed():
    langs = topica.stopword_languages()
    assert isinstance(langs, list)
    assert langs == sorted(langs)
    # a spread of families is present
    for code in ("en", "fr", "de", "es", "pt", "zh", "ar", "ru", "ja"):
        assert code in langs


def test_stopwords_by_code():
    fr = topica.stopwords("fr")
    assert isinstance(fr, frozenset)
    assert "le" in fr and "et" in fr
    assert len(fr) > 100


def test_stopwords_by_name_and_case_insensitive():
    assert topica.stopwords("french") == topica.stopwords("fr")
    assert topica.stopwords("German") == topica.stopwords("de")
    assert topica.stopwords(" Spanish ") == topica.stopwords("es")


def test_iso_english_is_larger_than_default():
    # stopwords("en") is the comprehensive iso list; ENGLISH_STOPWORDS is the
    # short default. They are different objects with different sizes.
    assert len(topica.stopwords("en")) > len(topica.ENGLISH_STOPWORDS)


def test_unknown_language_raises_with_available_codes():
    with pytest.raises(ValueError, match="no bundled stopword list"):
        topica.stopwords("klingon")
    try:
        topica.stopwords("xx")
    except ValueError as e:
        assert "en" in str(e) and "fr" in str(e)  # lists the available codes


def test_plugs_into_corpus_builder():
    docs = [
        "le chat et le chien mangent",
        "la planete et la lune brillent",
    ] * 10
    tokenized = [topica.tokenize(t, stopwords=topica.stopwords("fr")) for t in docs]
    flat = {w for d in tokenized for w in d}
    # French function words dropped; content words kept.
    assert "le" not in flat and "et" not in flat and "la" not in flat
    assert "chat" in flat and "planete" in flat
