"""PT (pseudo-document / short text), PA (Pachinko), and HLDA (hierarchical)
models: structure recovery, output shapes, determinism, and save/load.
"""

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# PT — Pseudo-document Topic Model (short texts)
# ---------------------------------------------------------------------------

def _short_corpus():
    # Two disjoint vocabularies; each "document" is just 2 tokens (short text).
    return (
        [["cat", "dog"]] * 40 + [["cat", "pet"]] * 40
        + [["star", "moon"]] * 40 + [["moon", "sky"]] * 40
    )


class TestPT:
    def test_shapes_and_recovery(self):
        docs = _short_corpus()
        m = topica.PT(num_topics=2, num_pseudo=10, seed=1)
        m.fit(docs, iters=300)
        assert m.topic_word.shape == (2, len(m.vocabulary))
        assert m.doc_topic.shape == (len(docs), 2)
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
        # The two topics separate the two vocabularies.
        blocks = [{"cat", "dog", "pet"}, {"star", "moon", "sky"}]
        tops = [{w for w, _ in m.top_words(3, topic=t)} for t in range(2)]
        owned = [max(range(2), key=lambda b: len(tops[t] & blocks[b])) for t in range(2)]
        assert set(owned) == {0, 1}

    def test_deterministic(self):
        docs = _short_corpus()
        a = topica.PT(num_topics=2, num_pseudo=10, seed=3); a.fit(docs, iters=150)
        b = topica.PT(num_topics=2, num_pseudo=10, seed=3); b.fit(docs, iters=150)
        assert np.array_equal(a.topic_word, b.topic_word)

    def test_save_load(self, tmp_path):
        docs = _short_corpus()
        m = topica.PT(num_topics=2, num_pseudo=10, seed=1); m.fit(docs, iters=150)
        p = str(tmp_path / "pt.tt"); m.save(p)
        loaded = topica.PT.load(p)
        assert np.array_equal(m.topic_word, loaded.topic_word)

    def test_bad_params(self):
        with pytest.raises(ValueError):
            topica.PT(num_topics=1)
        with pytest.raises(ValueError):
            topica.PT(num_topics=2, num_pseudo=0)
        with pytest.raises(ValueError):
            # pseudo_doc_prior must be finite and positive (#481-class guard).
            topica.PT(num_topics=2, pseudo_doc_prior=0.0)

    def test_warns_when_num_pseudo_exceeds_num_docs(self):
        # PTM's regime is P << D; fitting with num_pseudo >= num_docs collapses
        # the pseudo-document aggregation toward per-document LDA and should warn
        # (not error). See issue #491.
        docs = [["cat", "dog"], ["star", "moon"], ["cat", "pet"]]
        m = topica.PT(num_topics=2, num_pseudo=10, seed=1)
        with pytest.warns(UserWarning, match="num_pseudo"):
            m.fit(docs, iters=20)

    def test_no_warning_in_normal_regime(self):
        # P << D: no aggregation warning.
        import warnings

        docs = _short_corpus()
        m = topica.PT(num_topics=2, num_pseudo=10, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.fit(docs, iters=20, keep_theta_draws=False)


# ---------------------------------------------------------------------------
# PA — Pachinko Allocation (super/sub topics)
# ---------------------------------------------------------------------------

def _grouped_corpus(seed=0):
    # 4 sub-topic blocks; super-topic 0 = {block0, block1}, super 1 = {block2, block3}.
    rng = np.random.default_rng(seed)
    blocks = [[f"b{g}w{i}" for i in range(5)] for g in range(4)]
    docs = []
    for _ in range(120):
        if rng.random() < 0.5:
            pair = (blocks[0], blocks[1])  # super-topic 0
        else:
            pair = (blocks[2], blocks[3])  # super-topic 1
        doc = []
        for blk in pair:
            doc += [blk[int(rng.integers(5))] for _ in range(6)]
        docs.append(doc)
    return docs


class TestPA:
    def test_shapes(self):
        docs = _grouped_corpus()
        m = topica.PA(num_super=2, num_sub=4, seed=1)
        m.fit(docs, iters=300)
        v = len(m.vocabulary)
        assert m.topic_word.shape == (4, v)
        assert m.doc_topic.shape == (len(docs), 4)
        assert m.super_sub.shape == (2, 4)
        assert m.num_super == 2 and m.num_sub == 4
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)

    def test_subtopics_cover_blocks(self):
        docs = _grouped_corpus()
        m = topica.PA(num_super=2, num_sub=4, seed=1)
        m.fit(docs, iters=400)
        # All four planted vocabulary blocks appear among the sub-topics' top
        # words (blocks within a super-topic always co-occur here, so we check
        # coverage rather than per-sub-topic purity).
        covered = set()
        for t in range(4):
            for w, _ in m.top_words(4, topic=t):
                covered.add(w.split("w")[0])
        assert {"b0", "b1", "b2", "b3"} <= covered

    def test_super_sub_well_formed(self):
        docs = _grouped_corpus()
        m = topica.PA(num_super=2, num_sub=4, seed=1)
        m.fit(docs, iters=300)
        # super_sub is a finite, non-negative (S, K) association matrix. (Clean
        # super-topic separation isn't guaranteed on small data — super-topics
        # are exchangeable — so the Rust core test checks grouping on calibrated
        # data; here we just check the binding exposes a valid matrix.)
        ss = m.super_sub
        assert ss.shape == (2, 4)
        assert np.all(np.isfinite(ss)) and np.all(ss >= 0)

    def test_save_load(self, tmp_path):
        docs = _grouped_corpus()
        m = topica.PA(num_super=2, num_sub=4, seed=1); m.fit(docs, iters=150)
        p = str(tmp_path / "pa.tt"); m.save(p)
        loaded = topica.PA.load(p)
        assert np.array_equal(m.topic_word, loaded.topic_word)
        assert np.array_equal(m.super_sub, loaded.super_sub)


# ---------------------------------------------------------------------------
# HLDA — Hierarchical LDA (nested CRP tree)
# ---------------------------------------------------------------------------

def _hierarchical_corpus():
    # Shared function words in every doc; plus one of two group vocabularies.
    shared = ["the", "of", "and"]
    g0 = ["cat", "dog", "pet"]
    g1 = ["star", "moon", "sky"]
    return [shared + g0] * 40 + [shared + g1] * 40


class TestHLDA:
    def test_tree_structure(self):
        docs = _hierarchical_corpus()
        m = topica.HLDA(depth=2, seed=1)
        m.fit(docs, iters=300)
        assert m.num_nodes >= 2
        assert len(m.node_levels) == m.num_nodes
        assert len(m.node_parents) == m.num_nodes
        assert len(m.doc_paths) == len(docs)
        # Root is level 0 with parent -1; every doc path starts at a level-0 node.
        assert 0 in m.node_levels
        roots = [i for i in range(m.num_nodes) if m.node_parents[i] == -1]
        assert len(roots) == 1
        for path in m.doc_paths:
            assert m.node_levels[path[0]] == 0
        # The leaves are the deepest nodes.
        assert all(m.node_levels[leaf] >= 1 for leaf in m.leaves)

    def test_root_captures_shared_words(self):
        docs = _hierarchical_corpus()
        m = topica.HLDA(depth=2, seed=1)
        m.fit(docs, iters=300)
        root = m.node_parents.index(-1)
        root_top = {w for w, _ in m.top_words(root, 3)}
        # The shared function words should dominate the root topic.
        assert len(root_top & {"the", "of", "and"}) >= 2

    def test_deterministic(self):
        docs = _hierarchical_corpus()
        a = topica.HLDA(depth=2, seed=5); a.fit(docs, iters=150)
        b = topica.HLDA(depth=2, seed=5); b.fit(docs, iters=150)
        assert a.num_nodes == b.num_nodes
        assert a.node_levels == b.node_levels

    def test_save_load(self, tmp_path):
        docs = _hierarchical_corpus()
        m = topica.HLDA(depth=2, seed=1); m.fit(docs, iters=150)
        p = str(tmp_path / "hlda.tt"); m.save(p)
        loaded = topica.HLDA.load(p)
        assert m.num_nodes == loaded.num_nodes
        assert np.array_equal(m.topic_word, loaded.topic_word)
        assert m.doc_paths == loaded.doc_paths

    def test_bad_depth(self):
        with pytest.raises(ValueError):
            topica.HLDA(depth=1)
