import numpy as np
import pytest
import topica
from topica.ensemble import cross_ensemble, EnsembleResult


class DummyModel:
    def __init__(self, topic_word, doc_topic=None, vocabulary=None):
        self.topic_word = np.asarray(topic_word, dtype=np.float64)
        self.doc_topic = np.asarray(doc_topic, dtype=np.float64) if doc_topic is not None else None
        self.vocabulary = list(vocabulary) if vocabulary is not None else None


def test_cross_ensemble_basic():
    # Model A: K=2, V=4
    # Model B: K=3, V=4
    # Sharing the same vocabulary and document count
    # Let's create dummy topic-word distributions
    phi_a = np.array([
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.2]
    ])
    theta_a = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ]) # 2 docs, 2 topics

    phi_b = np.array([
        [0.75, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.75, 0.25],
        [0.2, 0.2, 0.3, 0.3]
    ])
    theta_b = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1]
    ]) # 2 docs, 3 topics

    vocab = ["apple", "banana", "cherry", "date"]
    model_a = DummyModel(phi_a, theta_a, vocab)
    model_b = DummyModel(phi_b, theta_b, vocab)

    # We expect default num_topics = median([2, 3]) = 2
    res = cross_ensemble([model_a, model_b])

    assert isinstance(res, EnsembleResult)
    assert res.topic_word.shape == (2, 4)
    assert res.doc_topic.shape == (2, 2)
    assert res.vocabulary == vocab
    assert len(res.stability) == 2
    assert len(res.support) == 2
    assert res.n_runs == 2


def test_cross_ensemble_vocab_intersection():
    # Model A vocabulary: ["apple", "banana", "cherry", "date"]
    phi_a = np.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.7]
    ])
    model_a = DummyModel(phi_a, vocabulary=["apple", "banana", "cherry", "date"])

    # Model B vocabulary: ["banana", "cherry", "date", "elderberry"]
    phi_b = np.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.7]
    ])
    model_b = DummyModel(phi_b, vocabulary=["banana", "cherry", "date", "elderberry"])

    # Intersection: ["banana", "cherry", "date"]
    res = cross_ensemble([model_a, model_b], num_topics=2)

    assert res.vocabulary == ["banana", "cherry", "date"]
    assert res.topic_word.shape == (2, 3)
    # Check that rows sum to 1.0
    assert np.allclose(res.topic_word.sum(axis=1), 1.0)


def test_cross_ensemble_no_thetas():
    # Models without doc_topic matrices (e.g. raw matrices or models where it's None)
    phi_a = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.1, 0.8]
    ])
    model_a = DummyModel(phi_a, vocabulary=["apple", "banana", "cherry"])

    phi_b = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.1, 0.8]
    ])
    model_b = DummyModel(phi_b, vocabulary=["apple", "banana", "cherry"])

    # This should run and warn/default to lambda_=1.0 (since no thetas are present)
    with pytest.warns(UserWarning, match="Models do not all share document-topic"):
        res = cross_ensemble([model_a, model_b], lambda_=0.5)

    assert res.doc_topic is None
    assert res.topic_word.shape == (2, 3)


def test_cross_ensemble_singleton_scores_zero_stability():
    # Two topics are shared by all three models (corroborated); each model also
    # has one idiosyncratic topic no other model matches. Those land in singleton
    # clusters, which must score stability 0.0 -- not 1.0 -- so they never look
    # reliable and do not inflate the headline `agreement`.
    V = 10

    def topic(*idx):
        v = np.full(V, 1e-3)
        for i in idx:
            v[i] = 1.0
        return v / v.sum()

    vocab = [f"w{i}" for i in range(V)]
    shared = [topic(0, 1), topic(2, 3)]
    uniques = [topic(4, 5), topic(6, 7), topic(8, 9)]
    models = [DummyModel(np.vstack(shared + [u]), vocabulary=vocab) for u in uniques]

    with pytest.warns(UserWarning, match="single model"):
        res = cross_ensemble(models, num_topics=5, lambda_=1.0, topn=2)

    singleton = res.cluster_sizes == 1
    assert int(singleton.sum()) == 3                        # the three idiosyncratic topics
    assert np.allclose(res.stability[singleton], 0.0)        # 0.0, not a misleading 1.0
    assert not res.reliable[singleton].any()                 # a lone topic is never reliable
    assert int(res.reliable.sum()) == 2                      # only the two corroborated topics
    assert res.agreement < 0.5                               # singletons no longer inflate it


def test_cross_ensemble_single_model_multitopic_cluster_scores_zero():
    # A cluster can hold more than one topic yet come from a *single* model: give
    # each model two identical copies of its own idiosyncratic topic. Those copies
    # cluster together (distance 0), so the cluster has 2 members but only 1
    # contributing model -- vacuous corroboration. Keying stability on member
    # count would score it 1.0 and mark it reliable; keying on contributing runs
    # (the fix) scores it 0.0, marks it unreliable, and counts it as a singleton.
    V = 10

    def topic(*idx):
        v = np.full(V, 1e-3)
        for i in idx:
            v[i] = 1.0
        return v / v.sum()

    vocab = [f"w{i}" for i in range(V)]
    shared = topic(0, 1)
    # Model A: shared + two identical copies of a (4,5) topic; Model B: shared +
    # two identical copies of a (6,7) topic.
    model_a = DummyModel(np.vstack([shared, topic(4, 5), topic(4, 5)]), vocabulary=vocab)
    model_b = DummyModel(np.vstack([shared, topic(6, 7), topic(6, 7)]), vocabulary=vocab)

    with pytest.warns(UserWarning, match="single model"):
        res = cross_ensemble([model_a, model_b], num_topics=3, lambda_=1.0, topn=2)

    # Every cluster here has two members; what separates the vacuous ones is that
    # both members come from the *same* model (support 0.5, not 1.0).
    assert np.all(res.cluster_sizes == 2)
    same_model = res.support < 1.0                           # the two duplicated idiosyncratic pairs
    assert int(same_model.sum()) == 2
    assert np.allclose(res.support[same_model], 0.5)         # one of two models
    assert np.allclose(res.stability[same_model], 0.0)       # not the misleading 1.0 of identical copies
    assert not res.reliable[same_model].any()                # one model agreeing with itself is not consensus
    assert res.reliable[res.support == 1.0].all()            # the genuinely shared topic stays reliable


def test_cross_ensemble_validation_guards():
    # Case 1: Less than 2 models
    model = DummyModel([[1.0]], vocabulary=["apple"])
    with pytest.raises(ValueError, match="need at least two models"):
        cross_ensemble([model])

    # Case 2: Incompatible document count
    model_a = DummyModel([[1.0]], doc_topic=[[1.0]], vocabulary=["apple"])
    model_b = DummyModel([[1.0]], doc_topic=[[1.0], [1.0]], vocabulary=["apple"])
    with pytest.raises(ValueError, match="Models must share the same document count"):
        cross_ensemble([model_a, model_b])

    # Case 3: Document count does not match texts length
    model_a = DummyModel([[1.0]], doc_topic=[[1.0]], vocabulary=["apple"])
    model_b = DummyModel([[1.0]], doc_topic=[[1.0]], vocabulary=["apple"])
    with pytest.raises(ValueError, match="does not match length of texts"):
        cross_ensemble([model_a, model_b], texts=["doc1", "doc2"])

    # Case 4: No vocabulary intersection
    model_a = DummyModel([[1.0]], vocabulary=["apple"])
    model_b = DummyModel([[1.0]], vocabulary=["banana"])
    with pytest.raises(ValueError, match="share no common vocabulary terms"):
        cross_ensemble([model_a, model_b])


def test_cross_ensemble_input_hardening():
    # Guards surfaced by the whole-module audit: lambda_ out of range, non-finite
    # weights, empty (K == 0) models, and num_topics beyond the pooled topics.
    vocab = ["apple", "banana", "cherry"]
    a = DummyModel([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], vocabulary=vocab)
    b = DummyModel([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], vocabulary=vocab)

    with pytest.raises(ValueError, match=r"lambda_ must be in \[0, 1\]"):
        cross_ensemble([a, b], lambda_=2.0)
    with pytest.raises(ValueError, match="weights must all be finite"):
        cross_ensemble([a, b], weights=[np.nan, 1.0])
    with pytest.raises(ValueError, match="no topics"):
        cross_ensemble([DummyModel(np.empty((0, 3)), vocabulary=vocab),
                        DummyModel(np.empty((0, 3)), vocabulary=vocab)])

    # num_topics beyond the 4 pooled topics warns and returns what exists.
    with pytest.warns(UserWarning, match="exceeds"):
        res = cross_ensemble([a, b], num_topics=99, lambda_=1.0)
    assert res.topic_word.shape[0] == 4
