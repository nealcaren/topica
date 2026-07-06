import numpy as np
import pytest
import topica
from topica.validation import align_topics, AlignmentResult


class DummyModel:
    def __init__(self, topic_word, vocabulary):
        self.topic_word = np.asarray(topic_word, dtype=np.float64)
        self.vocabulary = list(vocabulary)


def test_align_topics_basic_cosine():
    # 2 topics, 3 words vocabulary
    # Topic 0: [0.8, 0.2, 0.0]
    # Topic 1: [0.1, 0.0, 0.9]
    # We compare with another fit (rotated or slightly perturbed)
    phi_a = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.0, 0.9]
    ])
    phi_b = np.array([
        [0.0, 0.1, 0.9], # matches Topic 1
        [0.7, 0.3, 0.0]  # matches Topic 0
    ])

    res = align_topics(phi_a, phi_b, metric="cosine")
    
    assert isinstance(res, AlignmentResult)
    assert isinstance(res, list)
    assert len(res) == 2
    
    # Check Hungarian 1-to-1 matching (returns sorted by topic_a)
    # topic_a 0 should map to topic_b 1
    # topic_a 1 should map to topic_b 0
    assert res[0][0] == 0 and res[0][1] == 1
    assert res[1][0] == 1 and res[1][1] == 0
    
    # Check similarity matrix shape
    assert res.similarity_matrix.shape == (2, 2)
    # Check 1-to-1 matches
    assert len(res.matches) == 2
    assert res.matches[0][0] == 0 and res.matches[0][1] == 1
    assert res.matches[1][0] == 1 and res.matches[1][1] == 0


def test_align_topics_js():
    phi_a = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.0, 0.9]
    ])
    phi_b = np.array([
        [0.0, 0.1, 0.9],
        [0.7, 0.3, 0.0]
    ])
    res = align_topics(phi_a, phi_b, metric="js")
    assert len(res) == 2
    assert res[0][0] == 0 and res[0][1] == 1
    assert res[1][0] == 1 and res[1][1] == 0


def test_align_topics_rbo():
    # RBO focuses on ranking of top words
    model_a = DummyModel(
        topic_word=[
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.3]
        ],
        vocabulary=["apple", "banana", "cherry", "date"]
    )
    model_b = DummyModel(
        topic_word=[
            [0.0, 0.0, 0.8, 0.2],
            [0.9, 0.1, 0.0, 0.0]
        ],
        vocabulary=["apple", "banana", "cherry", "date"]
    )

    res = align_topics(model_a, model_b, metric="rbo", depth=2)
    assert len(res) == 2
    # Topic 0 in model_a (apple, banana) matches Topic 1 in model_b (apple, banana)
    assert res[0][0] == 0 and res[0][1] == 1
    # Topic 1 in model_a (cherry, date) matches Topic 0 in model_b (cherry, date)
    assert res[1][0] == 1 and res[1][1] == 0


def test_align_topics_emd():
    model_a = DummyModel(
        topic_word=[
            [0.9, 0.1, 0.0],
            [0.0, 0.1, 0.9]
        ],
        vocabulary=["apple", "banana", "cherry"]
    )
    model_b = DummyModel(
        topic_word=[
            [0.0, 0.2, 0.8],
            [0.85, 0.15, 0.0]
        ],
        vocabulary=["apple", "banana", "cherry"]
    )

    # Test EMD with binary cost (no embeddings)
    res = align_topics(model_a, model_b, metric="emd", depth=3)
    assert len(res) == 2
    assert res[0][0] == 0 and res[0][1] == 1
    assert res[1][0] == 1 and res[1][1] == 0

    # Test EMD with dictionary embeddings
    # apple and banana are close, cherry is far
    embeddings = {
        "apple": np.array([1.0, 0.0]),
        "banana": np.array([0.9, 0.1]),
        "cherry": np.array([0.0, 1.0])
    }
    res_emb = align_topics(model_a, model_b, metric="emd", depth=3, word_embeddings=embeddings)
    assert len(res_emb) == 2
    assert res_emb[0][0] == 0 and res_emb[0][1] == 1
    assert res_emb[1][0] == 1 and res_emb[1][1] == 0

    # Test EMD with ndarray embeddings (aligned to common_vocab)
    emb_matrix = np.array([
        [1.0, 0.0], # apple
        [0.9, 0.1], # banana
        [0.0, 1.0]  # cherry
    ])
    res_matrix = align_topics(model_a, model_b, metric="emd", depth=3, word_embeddings=emb_matrix)
    assert len(res_matrix) == 2
    assert res_matrix[0][0] == 0 and res_matrix[0][1] == 1
    assert res_matrix[1][0] == 1 and res_matrix[1][1] == 0


def test_align_topics_vocab_intersection():
    # Model A has vocabulary: ["apple", "banana", "cherry", "date"]
    model_a = DummyModel(
        topic_word=[
            [0.7, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.7]
        ],
        vocabulary=["apple", "banana", "cherry", "date"]
    )
    # Model B has vocabulary: ["banana", "cherry", "date", "elderberry"]
    model_b = DummyModel(
        topic_word=[
            [0.1, 0.2, 0.7, 0.0], # date, cherry, banana, elderberry
            [0.7, 0.2, 0.1, 0.0]
        ],
        vocabulary=["date", "cherry", "banana", "elderberry"]
    )

    # Intersection will be: ["banana", "cherry", "date"]
    res = align_topics(model_a, model_b, metric="cosine")
    # Model A Topic 0: main words banana, cherry -> matches Model B Topic 0 (banana, cherry)
    # Model A Topic 1: main words cherry, date -> matches Model B Topic 1 (date, cherry)
    assert len(res) == 2
    assert res[0][0] == 0 and res[0][1] == 0
    assert res[1][0] == 1 and res[1][1] == 1


def test_align_topics_splits_merges_unaligned():
    # We construct a scenario with splits, merges, and unaligned topics
    # Model A has 3 topics, Model B has 3 topics
    # A0 matches B0 (sim=0.9) - clean 1-to-1
    # A1 matches B1 (sim=0.8) and B2 (sim=0.7) - split
    # B2 matches A1 (sim=0.7) and A2 (sim=0.6) - merge (B2 merges from A1 and A2)
    # A2 also matches B2 (sim=0.6) and is unaligned to others
    # Let's specify threshold=0.5
    
    # We will build topic-word distributions to represent this
    # Let's construct a synthetic similarity matrix directly and check the classification logic
    # For a robust unit test, let's test our classification logic on top_word arrays
    # Topic A0: [1.0, 0.0, 0.0, 0.0]
    # Topic A1: [0.0, 1.0, 0.0, 0.0]
    # Topic A2: [0.0, 0.0, 1.0, 0.0]
    # Topic A3: [0.0, 0.0, 0.0, 1.0] (completely unaligned in B)
    phi_a = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ])
    # Topic B0 matches A0
    # Topic B1 matches A1 and A2 (split/merge)
    # Topic B2 matches A1 (split)
    # Topic B3 is completely unaligned in A
    phi_b = np.array([
        [0.9, 0.0, 0.0, 0.0, 0.1], # matches A0
        [0.0, 0.6, 0.6, 0.0, 0.0], # matches A1 and A2
        [0.0, 0.7, 0.0, 0.0, 0.3], # matches A1
        [0.0, 0.0, 0.0, 0.0, 1.0]  # matches nothing in A
    ])
    
    res = align_topics(phi_a, phi_b, threshold=0.5)
    
    # Matches: (0, 0) is a clean 1-to-1 match (A0 only matches B0, B0 only matches A0)
    assert (0, 0) in [(m[0], m[1]) for m in res.matches]
    
    # Splits: A1 matches B1 (sim=0.707) and B2 (sim=0.7)
    # Let's verify that A1 is classified as a split
    assert 1 in res.splits
    # A1 splits into B1 and B2
    targets_a1 = [t[0] for t in res.splits[1]]
    assert 1 in targets_a1
    assert 2 in targets_a1

    # Merges: B1 matches A1 (sim=0.707) and A2 (sim=0.707)
    assert 1 in res.merges
    sources_b1 = [s[0] for s in res.merges[1]]
    assert 1 in sources_b1
    assert 2 in sources_b1

    # Unaligned: A3 matches nothing (sim=0.0)
    assert 3 in res.unaligned_a
    # B3 matches nothing
    assert 3 in res.unaligned_b
