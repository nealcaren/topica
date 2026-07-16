"""Tests for the post-hoc topic diagnostics added alongside contrastive_topics:
topic_dendrogram (multi-resolution merge tree), flag_topics (per-topic junk
flag), and document_residuals (per-document novelty/coverage).
"""

import numpy as np
import pytest

import topica


class _FakeModel:
    """Minimal stand-in exposing the attributes the diagnostics read, so the
    dendrogram can be exercised on a hand-built, deterministic topic-word matrix
    instead of a randomly-fit model."""

    def __init__(self, topic_word, vocabulary, doc_topic=None):
        self.topic_word = np.asarray(topic_word, dtype=float)
        self.vocabulary = list(vocabulary)
        self.doc_topic = doc_topic

    @property
    def num_topics(self):
        return self.topic_word.shape[0]


def _paired_topics():
    """Six topics arranged as three near-duplicate pairs over a 12-word vocab:
    (0,1) live on words 0-2, (2,3) on 4-6, (4,5) on 8-10. Pairs differ only
    slightly within, so a 3-way cut should recover the pairing."""
    v = 12
    beta = np.full((6, v), 1e-3)
    blocks = [(0, [0, 1, 2]), (1, [0, 1, 2]),
              (2, [4, 5, 6]), (3, [4, 5, 6]),
              (4, [8, 9, 10]), (5, [8, 9, 10])]
    for t, words in blocks:
        for rank, w in enumerate(words):
            beta[t, w] = 1.0 + 0.01 * (rank + (t % 2))  # tiny within-pair tilt
    beta /= beta.sum(1, keepdims=True)
    vocab = [f"w{i}" for i in range(v)]
    return _FakeModel(beta, vocab)


class TestTopicDendrogram:
    def test_cut_recovers_paired_structure(self):
        pytest.importorskip("scipy")
        dnd = topica.topic_dendrogram(_paired_topics(), metric="js")
        labels = dnd.cut(3)
        # Each designed pair shares a label; the three pairs are distinct.
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[4] == labels[5]
        assert len({labels[0], labels[2], labels[4]}) == 3

    def test_merge_candidates_flags_near_duplicates(self):
        pytest.importorskip("scipy")
        dnd = topica.topic_dendrogram(_paired_topics(), metric="js")
        pairs = {(i, j) for i, j, _ in dnd.merge_candidates()}
        assert {(0, 1), (2, 3), (4, 5)} <= pairs
        # No cross-block pair should be flagged as a near-duplicate.
        assert (0, 2) not in pairs and (0, 4) not in pairs

    def test_groups_returns_members_and_words(self):
        pytest.importorskip("scipy")
        dnd = topica.topic_dendrogram(_paired_topics(), metric="js")
        groups = dnd.groups(3, n=3)
        assert len(groups) == 3
        all_members = sorted(m for members, _ in groups.values() for m in members)
        assert all_members == list(range(6))
        for _, words in groups.values():
            assert len(words) == 3

    def test_metrics_run(self):
        pytest.importorskip("scipy")
        model = _paired_topics()
        model.doc_topic = np.eye(6)[np.repeat(np.arange(6), 4)]  # for doctopic
        for metric in ("js", "hellinger", "cosine", "doctopic"):
            dnd = topica.topic_dendrogram(model, metric=metric)
            assert dnd.linkage.shape == (5, 4)
            assert dnd.distances.shape == (6, 6)

    def test_bad_metric_raises(self):
        pytest.importorskip("scipy")
        with pytest.raises(ValueError, match="metric must be"):
            topica.topic_dendrogram(_paired_topics(), metric="nope")


def _two_block_corpus(seed=0, soup=False):
    """A corpus with two clean lexical themes. When soup=True, every document is
    padded with stopwords so some fitted topics become boilerplate."""
    rng = np.random.default_rng(seed)
    a = ["tax", "budget", "economy", "growth", "jobs", "market", "trade"]
    b = ["troops", "war", "policy", "treaty", "border", "embassy", "summit"]
    stop = ["the", "of", "to", "and", "in", "is", "it", "that", "for", "on"]
    docs = []
    for _ in range(120):
        base_a = list(rng.choice(a, 12))
        base_b = list(rng.choice(b, 12))
        if soup:
            base_a = base_a[:4] + list(rng.choice(stop, 12))
            base_b = base_b[:4] + list(rng.choice(stop, 12))
        docs.append(base_a)
        docs.append(base_b)
    return docs


class TestFlagTopics:
    def test_fields_present(self):
        docs = _two_block_corpus()
        m = topica.models.LDA(num_topics=6, seed=0)
        m.fit(docs, iters=200)
        rows = topica.flag_topics(m, docs)
        assert len(rows) == 6
        keys = {"topic", "coherence", "exclusivity", "beta_entropy", "prevalence",
                "stopword_frac", "junk", "reasons", "top_words"}
        assert all(keys <= set(r) for r in rows)

    def test_catches_stopword_soup(self):
        soup = _two_block_corpus(soup=True)
        m = topica.models.LDA(num_topics=8, seed=0)
        m.fit(soup, iters=200)
        rows = topica.flag_topics(m, soup)
        soupy = [r for r in rows if "stopword-soup" in r["reasons"]]
        assert soupy, "expected at least one stopword-soup topic on an uncleaned corpus"

    def test_clean_corpus_has_no_stopword_flags(self):
        clean = _two_block_corpus(soup=False)
        m = topica.models.LDA(num_topics=6, seed=0)
        m.fit(clean, iters=200)
        rows = topica.flag_topics(m, clean)
        assert not any("stopword-soup" in r["reasons"] for r in rows)


class TestDocumentResiduals:
    def _model_with_injections(self, n_inject=12, seed=0):
        rng = np.random.default_rng(seed)
        on = ["tax", "budget", "economy", "growth", "jobs", "market",
              "troops", "war", "policy", "treaty", "border", "summit"]
        off = ["touchdown", "quarterback", "skillet", "saute", "oregano",
               "racquet", "volley", "puck", "colander", "ladle"]
        clean = [list(rng.choice(on, 14)) for _ in range(150)]
        injected = [list(rng.choice(off, 14)) for _ in range(n_inject)]
        docs = clean + injected
        inject_idx = set(range(len(clean), len(docs)))
        m = topica.models.LDA(num_topics=8, seed=0)
        m.fit(docs, iters=250)
        return m, docs, inject_idx

    def test_injected_offtopic_docs_rank_at_top(self):
        m, docs, inject_idx = self._model_with_injections(n_inject=12)
        res = topica.document_residuals(m, docs)
        k = len(inject_idx)
        top_k = {r["doc"] for r in res[:k]}
        precision = len(top_k & inject_idx) / k
        assert precision >= 0.8, f"precision@{k} = {precision}"

    def test_oov_tokens_raise_novelty(self):
        # OOV only arises when a scored document carries tokens outside the
        # fitted vocabulary. Append guaranteed-OOV tokens and check both the oov
        # fraction and the novelty rise relative to the unmodified documents.
        rng = np.random.default_rng(1)
        on = ["tax", "budget", "economy", "growth", "jobs", "market",
              "troops", "war", "policy", "treaty", "border", "summit"]
        clean = [list(rng.choice(on, 14)) for _ in range(120)]
        m = topica.models.LDA(num_topics=6, seed=0)
        m.fit(clean, iters=200)
        base = {r["doc"]: r for r in topica.document_residuals(m, clean)}
        with_oov = [d + ["zzqq1", "zzqq2", "zzqq3"] for d in clean]
        scored = {r["doc"]: r for r in topica.document_residuals(m, with_oov)}
        assert all(scored[d]["oov"] > 0 for d in range(len(clean)))
        assert (np.mean([scored[d]["novelty"] for d in range(len(clean))])
                > np.mean([base[d]["novelty"] for d in range(len(clean))]))

    def test_sorted_descending_and_fields(self):
        m, docs, _ = self._model_with_injections()
        res = topica.document_residuals(m, docs)
        nov = [r["novelty"] for r in res]
        assert nov == sorted(nov, reverse=True)
        keys = {"doc", "novelty", "cross_entropy", "kl", "cosine_dist",
                "oov", "n_tokens", "n_invocab"}
        assert all(keys <= set(r) for r in res)

    def test_rejects_misaligned_docs(self):
        m, docs, _ = self._model_with_injections()
        with pytest.raises(ValueError, match="doc_topic has"):
            topica.document_residuals(m, docs[:-3])
