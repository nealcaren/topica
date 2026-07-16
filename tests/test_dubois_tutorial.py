"""Fast guard for the Du Bois *Crisis* tutorial corpus.

Skips cleanly when ``examples/dubois_crisis.csv`` is absent (it is built by the
corpus builder, not checked into every clone). When present, it verifies the
CSV's shape and that the corpus is actually model-ready by fitting one tiny LDA
on a small subsample. Kept deliberately small so it finishes in a second or two.
"""

import csv
import os

import pytest

from topica.models import LDA
from topica import Corpus, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV_PATH = os.path.join(REPO, "examples", "dubois_crisis.csv")
STOP_PATH = os.path.join(REPO, "examples", "english-stoplist.txt")

EXPECTED_COLUMNS = [
    "title", "year", "decade", "volume", "issue", "author", "subjects", "text",
]

pytestmark = pytest.mark.skipif(
    not os.path.exists(CSV_PATH),
    reason="examples/dubois_crisis.csv not built (run the corpus builder first)",
)


def _load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EXPECTED_COLUMNS, reader.fieldnames
        return list(reader)


def test_csv_shape_and_columns():
    rows = _load_rows()
    assert len(rows) > 300, f"expected >300 articles, got {len(rows)}"
    # year/decade parse as ints; decade is the year's decade.
    for r in rows[:50]:
        year = int(r["year"])
        decade = int(r["decade"])
        assert 1900 <= year <= 1960
        assert decade == (year // 10) * 10
        assert r["text"].strip(), "empty text field"


def test_corpus_is_model_ready():
    """Fit one tiny LDA (K=3, 50 iters) on a 100-doc subsample."""
    rows = _load_rows()[:100]
    stopwords = sorted(set(open(STOP_PATH, encoding="utf-8").read().split()))
    docs = [tokenize(r["text"], stopwords=stopwords, min_length=3) for r in rows]
    assert all(len(d) > 0 for d in docs), "tokenization produced an empty doc"

    corpus = Corpus.from_documents(docs, min_doc_freq=3, rm_top=10)
    assert corpus.num_docs == len(docs)
    assert corpus.num_words > 50, f"vocab too small: {corpus.num_words}"

    lda = LDA(num_topics=3, seed=0)
    lda.fit(corpus, iters=50, num_samples=1, sample_interval=5)
    assert lda.num_topics == 3

    topics = lda.top_words(5)
    assert len(topics) == 3
    for words in topics:
        assert len(words) == 5
        for word, prob in words:
            assert isinstance(word, str) and word
            assert 0.0 <= prob <= 1.0
