"""Tests for ``topica.compare`` — statistical two-fit topic comparison (#415)."""

import numpy as np
import pytest

import topica
from topica.compare import CompareResult, MatchedPair, UnmatchedTopic


class DummyModel:
    """Minimal duck-typed fit: a topic-word matrix, a doc-topic matrix, a vocab."""

    def __init__(self, topic_word, doc_topic=None, vocabulary=None):
        self.topic_word = np.asarray(topic_word, dtype=np.float64)
        k, v = self.topic_word.shape
        if doc_topic is None:
            # uniform-ish doc-topic so prevalence is well-defined and non-degenerate
            doc_topic = np.tile(np.eye(k), (3, 1))  # (3k, k)
        self.doc_topic = np.asarray(doc_topic, dtype=np.float64)
        self.vocabulary = list(vocabulary) if vocabulary is not None else [
            f"w{i}" for i in range(v)
        ]


def _onehot_topics(k, v, seed=0, noise=0.0):
    """k topics, each concentrated on its own disjoint block of the vocab."""
    rng = np.random.default_rng(seed)
    phi = np.full((k, v), 1e-6)
    block = v // k
    for t in range(k):
        phi[t, t * block:(t + 1) * block] = 1.0
        if noise:
            phi[t] += rng.random(v) * noise
    return phi / phi.sum(axis=1, keepdims=True)


# --- alignment basics ---------------------------------------------------------

def test_compare_matches_permuted_topics():
    phi = _onehot_topics(4, 16, seed=1)
    a = DummyModel(phi)
    b = DummyModel(phi[[2, 0, 3, 1]])  # a known permutation
    cmp = topica.compare(a, b)
    assert isinstance(cmp, CompareResult)
    assert len(cmp.aligned) == 4
    assert not cmp.unmatched_a and not cmp.unmatched_b
    mapping = {p.topic_a: p.topic_b for p in cmp.aligned}
    assert mapping == {0: 1, 1: 3, 2: 0, 3: 2}
    assert all(p.similarity > 0.99 for p in cmp.aligned)
    assert all(isinstance(p, MatchedPair) for p in cmp.aligned)


def test_compare_prevalence_shift_point_estimate():
    phi = _onehot_topics(3, 12, seed=2)
    # A: topic 0 dominates; B: topic 0 halved, mass moved to topic 1.
    dt_a = np.array([[0.6, 0.2, 0.2]] * 10)
    dt_b = np.array([[0.3, 0.5, 0.2]] * 10)
    a = DummyModel(phi, doc_topic=dt_a)
    b = DummyModel(phi, doc_topic=dt_b)
    cmp = topica.compare(a, b)
    shifts = {(p.topic_a, p.topic_b): p.prevalence_shift for p in cmp.aligned}
    assert shifts[(0, 0)] == pytest.approx(-0.3, abs=1e-9)
    assert shifts[(1, 1)] == pytest.approx(+0.3, abs=1e-9)
    # DummyModel has no posterior over theta, so no SE (honestly None).
    assert all(p.prevalence_shift_se is None for p in cmp.aligned)


# --- honest unmatched / split at different K ---------------------------------

def test_compare_appeared_topic_at_higher_k():
    a = DummyModel(_onehot_topics(3, 12, seed=3))
    # B has the same 3 blocks plus a 4th disjoint block → one appeared topic.
    phi_b = np.full((4, 12), 1e-6)
    block = 12 // 3
    for t in range(3):
        phi_b[t, t * block:(t + 1) * block] = 1.0
    phi_b[3, 0:2] = 0.0
    phi_b[3, 10:12] = 1.0  # a distinct topic on the vocab tail
    phi_b = phi_b / phi_b.sum(axis=1, keepdims=True)
    b = DummyModel(phi_b)
    cmp = topica.compare(a, b, threshold=0.5)
    # Honesty invariant: every B topic is accounted for exactly once (matched,
    # a split target, or unmatched=appeared) and no A topic is force-paired away.
    b_matched = {p.topic_b for p in cmp.aligned}
    b_split = {j for v in cmp.splits.values() for j in v}
    b_appeared = {u.topic for u in cmp.unmatched_b}
    assert b_matched | b_split | b_appeared == {0, 1, 2, 3}
    # B's novel 4th topic is surfaced (as an appeared topic or a split target),
    # never silently dropped.
    assert 3 in (b_split | b_appeared)
    assert all(u.status == "appeared" for u in cmp.unmatched_b)
    assert all(u.side == "b" for u in cmp.unmatched_b)


# --- drift needs a null ------------------------------------------------------

def test_compare_no_null_reports_unknown_drift():
    phi = _onehot_topics(3, 12, seed=4)
    cmp = topica.compare(DummyModel(phi), DummyModel(phi))
    assert cmp.baseline["kind"] == "none"
    assert all(p.drifted is None for p in cmp.aligned)
    assert cmp.n_drifted is None


def test_compare_reseed_null_flags_only_the_moved_topic():
    v = 20
    phi_a = _onehot_topics(4, v, seed=5)
    a = DummyModel(phi_a)

    # Reseeds of A: tiny perturbations → tight self-agreement floor (~1.0).
    def refit(s):
        return DummyModel(_onehot_topics(4, v, seed=100 + s, noise=0.02))

    # B: topics 0-2 identical to A; topic 3 re-shapes *within its own block*
    # (A spreads mass over words 15-19; B spikes it on word 15). It still best-
    # matches A's topic 3 — so it stays a matched pair — but at a lower cosine than
    # any reseed of A achieves, which is exactly "drifted beyond reseed noise".
    phi_b = phi_a.copy()
    phi_b[3] = 1e-6
    phi_b[3, 15] = 1.0  # spike within topic 3's own block [15:20]
    phi_b[3] = phi_b[3] / phi_b[3].sum()
    b = DummyModel(phi_b)

    cmp = topica.compare(a, b, refit=refit, n_reseed=3)
    assert cmp.baseline["kind"] == "reseed"
    drift = {p.topic_a: p.drifted for p in cmp.aligned}
    # Topic 3 stayed matched (3→3) but drifted; the others are within reseed noise.
    assert (3, 3) in {(p.topic_a, p.topic_b) for p in cmp.aligned}
    assert drift.get(3) is True
    assert sum(1 for val in drift.values() if val) == 1
    assert cmp.n_drifted == 1


def test_compare_baseline_float_threshold():
    phi = _onehot_topics(3, 12, seed=6)
    a = DummyModel(phi)
    b = DummyModel(phi)
    # A flat, impossibly-high floor forces every (sim<floor) pair to "drifted".
    cmp = topica.compare(a, b, baseline=1.01)
    assert cmp.baseline == {"kind": "baseline", "similarity_floor": 1.01}
    assert all(p.drifted is True for p in cmp.aligned)
    # A floor below every similarity flags nothing.
    cmp2 = topica.compare(a, b, baseline=0.0)
    assert all(p.drifted is False for p in cmp2.aligned)


def test_compare_reseed_fits_list():
    phi = _onehot_topics(3, 12, seed=7)
    a = DummyModel(phi)
    reseeds = [DummyModel(_onehot_topics(3, 12, seed=200 + i, noise=0.02)) for i in range(3)]
    cmp = topica.compare(a, DummyModel(phi), reseed_fits=reseeds)
    assert cmp.baseline["kind"] == "reseed"
    assert all(p.drifted is False for p in cmp.aligned)  # identical → within noise


def test_compare_refit_callable_builds_the_null():
    # The refit= callable path (fast, dummy models): compare must call it n_reseed
    # times and derive the same kind of reseed floor the reseed_fits= list does.
    phi = _onehot_topics(3, 12, seed=11)
    a = DummyModel(phi)
    calls = []

    def refit(s):
        calls.append(s)
        return DummyModel(_onehot_topics(3, 12, seed=300 + s, noise=0.02))

    cmp = topica.compare(a, DummyModel(phi), refit=refit, n_reseed=4)
    assert len(calls) == 4  # called exactly n_reseed times
    assert cmp.baseline == {"kind": "reseed", "n_reseed": 4}
    assert all(p.drifted is False for p in cmp.aligned)  # identical → within noise


def test_compare_rejects_multiple_null_sources():
    a = DummyModel(_onehot_topics(2, 8, seed=8))
    with pytest.raises(ValueError, match="at most one drift null"):
        topica.compare(a, a, baseline=0.5, refit=lambda s: a)


# --- rendering / serialization ----------------------------------------------

def test_compare_render_html_and_markdown_and_dict():
    phi = _onehot_topics(3, 12, seed=9)
    cmp = topica.compare(DummyModel(phi), DummyModel(phi[[1, 2, 0]]), baseline=0.5)
    html = cmp.render()
    assert "topica-compare" in html and "Matched topics" in html
    md = cmp.to_markdown()
    assert md.startswith("# Topic comparison")
    assert "| A | B | similarity" in md
    d = cmp.to_dict()
    assert set(d) >= {"aligned", "unmatched_a", "unmatched_b", "splits", "merges", "baseline"}
    assert len(d["aligned"]) == 3

    # render(path=...) writes the file.
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "card.html")
        cmp.render(path=p)
        assert os.path.exists(p) and "topica-compare" in open(p).read()


def test_compare_public_api_exported():
    for name in ("compare", "CompareResult", "MatchedPair", "UnmatchedTopic"):
        assert hasattr(topica, name), f"topica.{name} not exported"
        assert name in topica.__all__, f"{name} missing from __all__"
    assert callable(topica.compare)


# --- manifest-native compare (#415): compare two provenance records -----------


def _manifest(top_words, prevalence=None):
    """A minimal AnalysisManifest carrying only what the manifest compare path
    reads: retained per-topic top words and (optionally) prevalence."""
    from topica import AnalysisManifest

    model = {"class": "X", "num_topics": len(top_words),
             "top_words": [list(t) for t in top_words]}
    if prevalence is not None:
        model["topic_prevalence"] = list(prevalence)
    return AnalysisManifest(topica_version="t", environment={}, model=model, corpus={})


def test_compare_manifests_matches_permuted_topics():
    # Four disjoint top-word sets, permuted in B → four clean Jaccard matches.
    tw = [["a", "b", "c", "d"], ["e", "f", "g", "h"],
          ["i", "j", "k", "l"], ["m", "n", "o", "p"]]
    a = _manifest(tw, prevalence=[0.4, 0.3, 0.2, 0.1])
    b = _manifest([tw[2], tw[0], tw[3], tw[1]], prevalence=[0.2, 0.4, 0.1, 0.3])
    cmp = topica.compare(a, b)
    assert cmp.metric == "jaccard"
    assert {p.topic_a: p.topic_b for p in cmp.aligned} == {0: 1, 1: 3, 2: 0, 3: 2}
    assert not cmp.unmatched_a and not cmp.unmatched_b
    assert all(p.similarity == 1.0 for p in cmp.aligned)
    # Prevalence carries through, shift is B minus A; no SE (no posterior stored).
    p0 = next(p for p in cmp.aligned if p.topic_a == 0)
    assert p0.prevalence_a == pytest.approx(0.4)
    assert p0.prevalence_b == pytest.approx(0.4)  # B topic 1 also 0.4
    assert all(p.prevalence_shift_se is None for p in cmp.aligned)


def test_compare_manifests_appeared_topic_at_higher_k():
    tw_a = [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]]
    tw_b = tw_a + [["q", "r", "s", "t"]]  # a genuinely new, disjoint topic
    cmp = topica.compare(_manifest(tw_a), _manifest(tw_b))
    b_matched = {p.topic_b for p in cmp.aligned}
    b_appeared = {u.topic for u in cmp.unmatched_b}
    assert 3 in b_appeared and 3 not in b_matched
    assert all(u.status == "appeared" and u.side == "b" for u in cmp.unmatched_b)


def test_compare_manifests_baseline_drift():
    tw = [["a", "b", "c"], ["e", "f", "g"]]
    a, b = _manifest(tw), _manifest(tw)
    hi = topica.compare(a, b, baseline=1.01)  # floor above every similarity
    assert hi.baseline == {"kind": "baseline", "similarity_floor": 1.01}
    assert all(p.drifted is True for p in hi.aligned)
    lo = topica.compare(a, b, baseline=0.0)
    assert all(p.drifted is False for p in lo.aligned)
    none = topica.compare(a, b)
    assert none.baseline["kind"] == "none"
    assert all(p.drifted is None for p in none.aligned)


def test_compare_manifests_missing_top_words_is_a_clear_error():
    from topica import AnalysisManifest

    good = _manifest([["a", "b"], ["c", "d"]])
    bare = AnalysisManifest(topica_version="t", environment={},
                            model={"class": "X", "num_topics": 2}, corpus={})
    with pytest.raises(ValueError, match="no retained top words"):
        topica.compare(bare, good)


def test_compare_rejects_model_manifest_mix():
    man = _manifest([["a", "b"], ["c", "d"]])
    live = DummyModel(_onehot_topics(2, 8, seed=1))
    with pytest.raises(TypeError, match="cannot compare a live model"):
        topica.compare(live, man)
    with pytest.raises(TypeError, match="cannot compare a live model"):
        topica.compare(man, live)


def test_compare_manifests_reject_refit_and_bad_metric():
    a = _manifest([["a", "b"], ["c", "d"]])
    b = _manifest([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError, match="cannot be refit"):
        topica.compare(a, b, refit=lambda s: a)
    with pytest.raises(ValueError, match="cannot be refit"):
        topica.compare(a, b, reseed_fits=[b])
    with pytest.raises(ValueError, match="metric must be 'jaccard'"):
        topica.compare(a, b, metric="cosine")


def test_compare_manifests_render_and_dict():
    tw = [["a", "b", "c"], ["e", "f", "g"]]
    cmp = topica.compare(_manifest(tw, [0.6, 0.4]), _manifest(tw, [0.5, 0.5]))
    assert "topica-compare" in cmp.render()
    assert cmp.to_markdown().startswith("# Topic comparison")
    d = cmp.to_dict()
    assert d["metric"] == "jaccard" and len(d["aligned"]) == 2


def test_compare_manifests_card_footer_omits_refit_options():
    # A manifest cannot be refit, so the "no null" card must advertise only
    # baseline= — never refit=/reseed_fits=, which would raise (real-user finding).
    cmp = topica.compare(_manifest([["a", "b"], ["c", "d"]]),
                         _manifest([["a", "b"], ["c", "d"]]))
    html = cmp.render()
    assert "baseline=" in html and "refit=" not in html and "reseed_fits=" not in html
    # The live path still offers all three.
    live = topica.compare(DummyModel(_onehot_topics(2, 8, seed=1)),
                          DummyModel(_onehot_topics(2, 8, seed=1)))
    assert "refit=" in live.render()


def test_compare_manifests_flags_near_miss_disappearance():
    # A stable topic whose top words churned enough to fall just under threshold is
    # reported vanished+appeared, but flagged near-miss so it is not silently false.
    a = [["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"],
         ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
    b = [["w0", "w1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"],  # shares 2/10 → J=0.111
         ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
    cmp = topica.compare(_manifest(a), _manifest(b))  # default jaccard threshold 0.12
    assert not any(p.topic_a == 0 for p in cmp.aligned)  # topic 0 fell below threshold
    va = next(u for u in cmp.unmatched_a if u.topic == 0)
    ap = next(u for u in cmp.unmatched_b if u.topic == 0)
    assert va.near_miss and ap.near_miss
    assert va.best_similarity == pytest.approx(2 / 18, abs=1e-6)
    assert "near-miss" in cmp.render() and "near-miss" in cmp.to_markdown()
    # A genuinely-disjoint vanished topic is NOT a near-miss.
    c = topica.compare(_manifest([["a", "b", "c"]]), _manifest([["x", "y", "z"]]))
    assert c.unmatched_a[0].best_similarity == 0.0
    assert not c.unmatched_a[0].near_miss


def test_compare_manifests_prevalence_length_mismatch_is_clear_error():
    from topica import AnalysisManifest

    bad = AnalysisManifest(
        topica_version="t", environment={},
        model={"class": "X", "num_topics": 3,
               "top_words": [["a"], ["b"], ["c"]], "topic_prevalence": [0.5, 0.5]},
        corpus={})
    with pytest.raises(ValueError, match="prevalence values but 3 topics"):
        topica.compare(bad, _manifest([["a"], ["b"], ["c"]]))


@pytest.mark.slow
def test_compare_manifests_agrees_with_live_on_real_lda():
    rng = np.random.default_rng(0)
    blocks = [[f"t{t}_w{i}" for i in range(15)] for t in range(6)]
    vocab_all = sum(blocks, [])
    docs = []
    for d in range(600):
        c = d % 6
        docs.append(list(rng.choice(blocks[c], size=12)) + list(rng.choice(vocab_all, size=3)))
    corpus = topica.Corpus.from_documents(docs)

    ma = topica.LDA(num_topics=6, seed=1)
    ma.fit(corpus, iters=200)
    mb = topica.LDA(num_topics=6, seed=2)
    mb.fit(corpus, iters=200)

    live = topica.compare(ma, mb)
    ra = topica.record_fit(ma, corpus, topic_words_n=10)
    rb = topica.record_fit(mb, corpus, topic_words_n=10)
    man = topica.compare(ra, rb)

    # On a real corpus with V ≫ top-N, top-word Jaccard recovers the same matching
    # the live cosine path finds — the manifest can stand in for the model.
    assert {(p.topic_a, p.topic_b) for p in man.aligned} == \
           {(p.topic_a, p.topic_b) for p in live.aligned}
    # Retained prevalence equals the live point estimate.
    for p in man.aligned:
        assert p.prevalence_a == pytest.approx(float(ma.doc_topic.mean(0)[p.topic_a]))


# --- integration with a real model -------------------------------------------

@pytest.mark.slow
def test_compare_real_lda_two_seeds_and_reseed_null():
    rng = np.random.default_rng(0)
    blocks = [
        ["alpha", "beta", "gamma", "delta"],
        ["red", "green", "blue", "cyan"],
        ["dog", "cat", "fish", "bird"],
        ["north", "south", "east", "west"],
    ]
    vocab_all = sum(blocks, [])
    docs = []
    for d in range(200):
        c = d % 4
        docs.append(list(rng.choice(blocks[c], size=8)) + list(rng.choice(vocab_all, size=2)))

    def fit(seed):
        m = topica.LDA(num_topics=4, seed=seed)
        m.fit(docs, iters=150)
        return m

    a, b = fit(1), fit(2)
    cmp = topica.compare(a, b, refit=fit, n_reseed=2)
    assert cmp.num_topics_a == cmp.num_topics_b == 4
    assert len(cmp.aligned) == 4
    # Two clean reseeds of the same planted corpus should not look like drift.
    assert cmp.n_drifted == 0
    # LDA is a Gibbs model with retained doc-lengths → prevalence carries an SE.
    assert all(p.prevalence_shift_se is not None for p in cmp.aligned)
    assert isinstance(cmp.render(), str)
