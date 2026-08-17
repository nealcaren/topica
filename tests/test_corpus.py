"""Tests for the Corpus class."""

import numpy as np
import pytest

from topica import Corpus


def test_doc_lengths_word_counts_are_numpy_arrays():
    # #775 T3.1: doc_lengths / word_counts are numpy arrays (not lists) on the
    # numpy-native corpus, so .mean()/.sum() work directly. vocabulary stays a list.
    docs = [["cat", "dog", "fish"], ["cat", "dog"], ["cat"]]
    c = Corpus.from_documents(docs)
    assert isinstance(c.doc_lengths, np.ndarray)
    assert isinstance(c.word_counts, np.ndarray)
    assert c.doc_lengths.tolist() == [3, 2, 1]
    assert float(c.doc_lengths.mean()) == pytest.approx(2.0)
    assert int(c.word_counts.sum()) == 6  # total token occurrences
    assert isinstance(c.vocabulary, list) and all(isinstance(w, str) for w in c.vocabulary)


# ---------------------------------------------------------------------------
# from_documents — basic construction
# ---------------------------------------------------------------------------

class TestFromDocumentsBasic:
    def test_num_docs(self):
        docs = [["cat", "dog"]] * 5 + [["planet", "star"]] * 5
        c = Corpus.from_documents(docs)
        assert c.num_docs == 10

    def test_num_words(self):
        docs = [["cat", "dog", "fish"]] * 3
        c = Corpus.from_documents(docs)
        assert c.num_words == 3

    def test_total_tokens(self):
        docs = [["cat", "dog", "fish"]] * 4   # 3 tokens × 4 docs = 12
        c = Corpus.from_documents(docs)
        assert c.total_tokens == 12

    def test_vocabulary_contents(self):
        docs = [["cat", "dog"], ["planet", "star"]]
        c = Corpus.from_documents(docs)
        assert sorted(c.vocabulary) == ["cat", "dog", "planet", "star"]

    def test_vocabulary_length_matches_num_words(self):
        docs = [["a", "b", "c"]] * 3
        c = Corpus.from_documents(docs)
        assert len(c.vocabulary) == c.num_words

    def test_doc_names_default_length(self):
        docs = [["cat"]] * 7
        c = Corpus.from_documents(docs)
        assert len(c.doc_names) == 7

    def test_doc_names_custom(self):
        docs = [["cat"], ["dog"]]
        c = Corpus.from_documents(docs, doc_names=["alice", "bob"])
        assert c.doc_names == ["alice", "bob"]

    def test_doc_labels_default_empty_strings(self):
        docs = [["cat"]] * 3
        c = Corpus.from_documents(docs)
        assert all(label == "" for label in c.doc_labels)

    def test_doc_labels_custom(self):
        docs = [["cat"], ["dog"]]
        c = Corpus.from_documents(docs, doc_labels=["A", "B"])
        assert c.doc_labels == ["A", "B"]


# ---------------------------------------------------------------------------
# from_documents — stopwords filtering
# ---------------------------------------------------------------------------

class TestStopwordsFiltering:
    def test_stopwords_removed_from_vocabulary(self):
        docs = [["cat", "dog", "the", "is"]] * 5
        c = Corpus.from_documents(docs, stopwords=["the", "is"])
        assert "the" not in c.vocabulary
        assert "is" not in c.vocabulary
        assert "cat" in c.vocabulary
        assert "dog" in c.vocabulary

    def test_stopwords_reduce_num_words(self):
        docs = [["cat", "dog", "the"]] * 5
        c_no_stop = Corpus.from_documents(docs)
        c_stop = Corpus.from_documents(docs, stopwords=["the"])
        assert c_stop.num_words == c_no_stop.num_words - 1


# ---------------------------------------------------------------------------
# from_documents — frequency filtering
# ---------------------------------------------------------------------------

class TestFrequencyFiltering:
    def test_min_doc_freq_filters_rare_word(self):
        docs = [
            ["cat", "dog", "rare"],  # "rare" only in this doc
            ["cat", "dog"],
            ["cat", "dog"],
        ]
        c = Corpus.from_documents(docs, min_doc_freq=2)
        assert "rare" not in c.vocabulary
        assert "cat" in c.vocabulary

    def test_min_doc_freq_keeps_common_word(self):
        docs = [["cat", "dog"]] * 5
        c = Corpus.from_documents(docs, min_doc_freq=3)
        assert "cat" in c.vocabulary
        assert "dog" in c.vocabulary

    def test_max_doc_fraction_filters_ubiquitous_word(self):
        # Use 100 docs: "cat" in 90/100 = 0.9; max_doc_fraction=0.89 → filtered
        docs = [["cat", "rare"]] * 90 + [["rare", "other"]] * 10
        c = Corpus.from_documents(docs, max_doc_fraction=0.89)
        assert "cat" not in c.vocabulary

    def test_max_doc_fraction_keeps_less_common_word(self):
        # "cat" in 90/100 = 0.9; max_doc_fraction=0.9 → kept (≤ threshold)
        docs = [["cat", "rare"]] * 90 + [["rare", "other"]] * 10
        c = Corpus.from_documents(docs, max_doc_fraction=0.9)
        assert "cat" in c.vocabulary


# ---------------------------------------------------------------------------
# from_documents — length mismatch raises ValueError
# ---------------------------------------------------------------------------

class TestLengthMismatch:
    def test_doc_names_mismatch_raises(self):
        docs = [["cat"]] * 5
        with pytest.raises(ValueError):
            Corpus.from_documents(docs, doc_names=["a", "b"])

    def test_doc_labels_mismatch_raises(self):
        docs = [["cat"]] * 5
        with pytest.raises(ValueError):
            Corpus.from_documents(docs, doc_labels=["x", "y"])


# ---------------------------------------------------------------------------
# save / load binary round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_num_docs(self, tmp_path):
        docs = [["cat", "dog"]] * 5 + [["planet", "star"]] * 5
        c = Corpus.from_documents(docs)
        path = str(tmp_path / "corpus.bin")
        c.save(path)
        c2 = Corpus.load(path)
        assert c2.num_docs == c.num_docs

    def test_roundtrip_preserves_num_words(self, tmp_path):
        docs = [["cat", "dog"]] * 5 + [["planet", "star"]] * 5
        c = Corpus.from_documents(docs)
        path = str(tmp_path / "corpus.bin")
        c.save(path)
        c2 = Corpus.load(path)
        assert c2.num_words == c.num_words

    def test_roundtrip_preserves_vocabulary(self, tmp_path):
        docs = [["cat", "dog"], ["planet", "star"]]
        c = Corpus.from_documents(docs)
        path = str(tmp_path / "corpus.bin")
        c.save(path)
        c2 = Corpus.load(path)
        assert sorted(c2.vocabulary) == sorted(c.vocabulary)

    def test_roundtrip_preserves_total_tokens(self, tmp_path):
        docs = [["cat", "dog", "fish"]] * 4
        c = Corpus.from_documents(docs)
        path = str(tmp_path / "corpus.bin")
        c.save(path)
        c2 = Corpus.load(path)
        assert c2.total_tokens == c.total_tokens


# ---------------------------------------------------------------------------
# from_text_file — plain format
# ---------------------------------------------------------------------------

class TestFromTextFilePlain:
    def test_plain_num_docs(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("cat dog fish\nplanet star moon\n")
        c = Corpus.from_text_file(str(p))
        assert c.num_docs == 2

    def test_plain_vocabulary(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("cat dog\nplanet star\n")
        c = Corpus.from_text_file(str(p))
        assert sorted(c.vocabulary) == ["cat", "dog", "planet", "star"]

    def test_plain_lowercases_tokens(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("Cat DOG\n")
        c = Corpus.from_text_file(str(p))
        vocab = c.vocabulary
        assert "cat" in vocab or "dog" in vocab  # at least one lowercased token
        assert "Cat" not in vocab
        assert "DOG" not in vocab


# ---------------------------------------------------------------------------
# from_text_file — tsv format
# ---------------------------------------------------------------------------

class TestFromTextFileTsv:
    def test_tsv_num_docs(self, tmp_path):
        p = tmp_path / "docs.tsv"
        p.write_text("doc1\tlabel1\tcat dog fish\ndoc2\tlabel2\tplanet star moon\n")
        c = Corpus.from_text_file(str(p), format="tsv")
        assert c.num_docs == 2

    def test_tsv_doc_names(self, tmp_path):
        p = tmp_path / "docs.tsv"
        p.write_text("doc1\tlabel1\tcat dog\ndoc2\tlabel2\tplanet star\n")
        c = Corpus.from_text_file(str(p), format="tsv")
        assert c.doc_names == ["doc1", "doc2"]

    def test_tsv_doc_labels(self, tmp_path):
        p = tmp_path / "docs.tsv"
        p.write_text("doc1\tlabelA\tcat dog\ndoc2\tlabelB\tplanet star\n")
        c = Corpus.from_text_file(str(p), format="tsv")
        assert c.doc_labels == ["labelA", "labelB"]

    def test_tsv_vocabulary(self, tmp_path):
        p = tmp_path / "docs.tsv"
        p.write_text("doc1\tlabel1\tcat dog\ndoc2\tlabel2\tplanet star\n")
        c = Corpus.from_text_file(str(p), format="tsv")
        assert sorted(c.vocabulary) == ["cat", "dog", "planet", "star"]


# ---------------------------------------------------------------------------
# from_text_file — bad format raises ValueError
# ---------------------------------------------------------------------------

class TestFromTextFileBadFormat:
    def test_bad_format_raises(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(ValueError):
            Corpus.from_text_file(str(p), format="xml")


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_num_docs(self):
        docs = [["cat"]] * 7
        c = Corpus.from_documents(docs)
        assert "7" in repr(c)

    def test_repr_contains_corpus(self):
        docs = [["cat"]]
        c = Corpus.from_documents(docs)
        assert "Corpus" in repr(c)


# ---------------------------------------------------------------------------
# preprocessing parameters (issue #399)
# ---------------------------------------------------------------------------

class TestPreprocessingGetter:
    def test_records_the_params_applied(self):
        c = Corpus.from_documents(
            [["a", "b", "c"], ["b", "c", "d"]],
            min_doc_freq=1, max_doc_fraction=1.0, min_cf=0, rm_top=2)
        assert c.preprocessing == {
            "min_doc_freq": 1, "max_doc_fraction": 1.0, "min_cf": 0, "rm_top": 2,
            "max_features": None, "vocabulary": False}

    def test_none_after_load(self, tmp_path):
        c = Corpus.from_documents([["a", "b"], ["b", "c", "a"]])
        assert c.preprocessing is not None
        p = tmp_path / "c.tt"
        c.save(str(p))
        # The save format does not carry the params, so a loaded corpus is honest
        # about not knowing them.
        assert Corpus.load(str(p)).preprocessing is None
