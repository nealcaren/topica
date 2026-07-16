"""Vocabulary filtering (min_cf / rm_top) on Corpus.from_documents and the
summary() overview helper."""

import topica

DOCS = [
    ["the", "cat", "sat"],
    ["the", "dog", "ran"],
    ["the", "cat", "ran"],
    ["rare", "word", "here"],
]


def test_no_filter_keeps_all():
    c = topica.Corpus.from_documents(DOCS)
    assert set(c.vocabulary) == {"the", "cat", "sat", "dog", "ran", "rare", "word", "here"}


def test_rm_top_drops_most_frequent():
    # "the" appears in 3 docs (3 times) — the single most frequent token.
    c = topica.Corpus.from_documents(DOCS, rm_top=1)
    assert "the" not in c.vocabulary
    assert "cat" in c.vocabulary


def test_min_cf_filters_by_total_frequency():
    # Keep words with collection frequency >= 2: the(3), cat(2), ran(2).
    c = topica.Corpus.from_documents(DOCS, min_cf=2)
    assert set(c.vocabulary) == {"the", "cat", "ran"}


def test_min_doc_freq_still_works():
    c = topica.Corpus.from_documents(DOCS, min_doc_freq=2)
    assert set(c.vocabulary) == {"the", "cat", "ran"}


def test_combined_filters():
    # rm_top removes "the"; min_cf=2 would have kept it, but rm_top wins.
    c = topica.Corpus.from_documents(DOCS, min_cf=2, rm_top=1)
    assert "the" not in c.vocabulary
    assert set(c.vocabulary) == {"cat", "ran"}


class TestSummary:
    def test_summary_has_topics_and_words(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = topica.models.LDA(num_topics=2, seed=1)
        m.fit(docs, iters=200)
        s = topica.interpret.summary(m, topn=3)
        assert "num_topics: 2" in s
        assert "vocab_size: 6" in s
        assert "topic 0:" in s and "topic 1:" in s

    def test_summary_graceful_for_dtm(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = topica.models.DTM(num_topics=2, seed=1)
        m.fit(docs, [0] * 20 + [1] * 20, iters=5)
        s = topica.interpret.summary(m)  # DTM.top_words needs (topic, time) -> per-topic omitted
        assert "num_times: 2" in s
        assert isinstance(s, str)
