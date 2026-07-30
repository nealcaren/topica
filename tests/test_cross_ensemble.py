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

    res = cross_ensemble(models, num_topics=5, lambda_=1.0, topn=2)

    singleton = res.cluster_sizes == 1
    assert int(singleton.sum()) == 3                        # the three idiosyncratic topics
    assert np.allclose(res.stability[singleton], 0.0)        # 0.0, not a misleading 1.0
    assert not res.reliable[singleton].any()                 # a lone topic is never reliable
    assert int(res.reliable.sum()) == 2                      # only the two corroborated topics
    assert res.agreement < 0.5                               # singletons no longer inflate it


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
