"""Statistical-equivalence check for topica's Top2Vec against BERTopic.

topica (in-house UMAP reducer) and BERTopic (umap-learn reducer) are independent
embedding-clustering topic models, so we do not expect identical topics. On a
controlled planted-cluster task with shared document embeddings, we require that
topica recovers the ground truth at least as well as BERTopic does (within a
margin) and that the two agree with each other. Skips cleanly when BERTopic and
its UMAP/HDBSCAN dependencies are unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import top2vec_compare  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not top2vec_compare.bertopic_available(),
    reason="BERTopic (with umap/hdbscan) not installed",
)
def test_top2vec_matches_bertopic():
    m = top2vec_compare.run(verbose=False)

    # Both implementations should recover the planted clusters well.
    assert m["topica_truth_ari"] >= 0.5, m
    # topica recovers the truth at least as well as BERTopic, minus a margin for
    # the PCA-vs-UMAP reducer difference.
    assert m["topica_truth_ari"] >= m["bertopic_truth_ari"] - 0.2, m
    # The two implementations broadly agree on the partition.
    assert m["cross_ari"] >= 0.4, m
    # Each topica topic's top c-TF-IDF words come from a single planted block.
    assert m["topica_block_purity"] >= 0.9, m
    # The discovered topic count is in the right ballpark (planted = 4).
    assert 2 <= m["topica_num_topics"] <= 8, m
