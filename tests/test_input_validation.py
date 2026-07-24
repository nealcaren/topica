"""Input-validation edge cases.

Adversarial inputs must fail loudly, not silently corrupt the fit:
  1. NaN/Inf float hyperparameters are rejected at construction.
  2. A corpus with no words (all documents empty) is rejected at fit.
  3. Misusing the diagnostics (raw matrix, non-integer topn) gives a clear error.
  4. coherence on an empty reference corpus errors rather than returning NaN.
  5. frex rejects out-of-range frequency weights.
  6. NaN/Inf in covariate/embedding matrices raises ValueError (issue #100).
  7. iters=0 and iters=-1 raise ValueError (issue #103).
"""

import numpy as np
import pytest

import topica
from topica import LDA

NAN = float("nan")
INF = float("inf")
DOCS = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"]] * 5


# --- 1. Non-finite hyperparameters -----------------------------------------

@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.0, -1.0])
def test_lda_rejects_nonfinite_beta(bad):
    with pytest.raises(ValueError):
        LDA(num_topics=2, beta=bad)


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, -1.0])
def test_lda_rejects_nonfinite_alpha_sum(bad):
    with pytest.raises(ValueError):
        LDA(num_topics=2, alpha_sum=bad)


def test_other_constructors_reject_nan():
    # A representative float hyperparameter on each model family.
    cases = [
        (topica.DMR, dict(num_topics=2, beta=NAN)),
        (topica.DMR, dict(num_topics=2, prior_variance=NAN)),
        (topica.LabeledLDA, dict(alpha=NAN)),
        (topica.HDP, dict(beta=NAN)),
        (topica.HDP, dict(alpha=NAN)),
        (topica.DTM, dict(num_topics=2, chain_variance=NAN)),
        (topica.SupervisedLDA, dict(num_topics=2, alpha=NAN)),
        (topica.SAGE, dict(num_topics=2, prior_variance=NAN)),
    ]
    for cls, kw in cases:
        with pytest.raises(ValueError):
            cls(**kw)


@pytest.mark.parametrize("bad", [NAN, INF, -0.01, 1.5])
def test_seededlda_rejects_bad_weight(bad):
    # #456: an unvalidated `weight` silently corrupted the seed prior — a negative
    # weight gives negative topic-word probabilities and NaN/inf gives non-finite
    # ones; the seededlda package restricts it to [0, 1].
    seeds = {"sports": ["ball", "goal"], "politics": ["vote", "law"]}
    with pytest.raises(ValueError, match="weight"):
        topica.SeededLDA(seeds, weight=bad)


@pytest.mark.parametrize("bad", [NAN, INF])
def test_seededlda_rejects_nonfinite_doc_topic_prior(bad):
    # #456: the doc_topic_prior guard was `a <= 0.0`, which is false for NaN and
    # +inf, so a non-finite prior slipped through and contaminated the fit. It must
    # be rejected up front (same non-finite class as the weight guard above).
    seeds = {"sports": ["ball", "goal"], "politics": ["vote", "law"]}
    docs = [["ball", "goal", "vote"], ["law", "vote", "ball"], ["goal", "law", "ball"]]
    m = topica.SeededLDA(seeds, seed=1)
    prior = [[1.0, bad] for _ in docs]
    with pytest.raises(ValueError, match="finite"):
        m.fit(docs, iters=10, doc_topic_prior=prior)


# #481: the `a <= 0.0` guard admits NaN/+inf (both compare false), so a
# non-finite constructor argument slipped past validation and produced NaN
# topic_word / doc_topic. Every strictly-positive param below now routes through
# ensure_finite_pos; the non-negative RTM params (rho/negative_ratio/ridge)
# through ensure_finite_nonneg. Both reject NaN and +inf.
_NONFINITE_POS_CASES = [
    ("BTM.alpha", lambda bad: topica.BTM(3, alpha=bad)),
    ("BTM.beta", lambda bad: topica.BTM(3, beta=bad)),
    ("PLTM.alpha", lambda bad: topica.PolylingualLDA(3, alpha=bad)),
    ("PLTM.beta", lambda bad: topica.PolylingualLDA(3, beta=bad)),
    ("RTM.alpha", lambda bad: topica.RTM(3, alpha=bad)),
    ("RTM.rho", lambda bad: topica.RTM(3, rho=bad)),
    ("RTM.negative_ratio", lambda bad: topica.RTM(3, negative_ratio=bad)),
    ("RTM.ridge", lambda bad: topica.RTM(3, ridge=bad)),
    ("FASTopic.theta_temp", lambda bad: topica.FASTopic(3, theta_temp=bad)),
]


@pytest.mark.parametrize("bad", [NAN, INF])
@pytest.mark.parametrize(
    "build", [c[1] for c in _NONFINITE_POS_CASES], ids=[c[0] for c in _NONFINITE_POS_CASES]
)
def test_nonfinite_hyperparameters_rejected_across_constructors(build, bad):
    with pytest.raises(ValueError, match="finite"):
        build(bad)


def test_tlda_nonfinite_hyperparameters_rejected():
    # TensorLDA is gated behind the experimental flag; enable it just for this check.
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        for kw in (dict(alpha_0=NAN), dict(learning_rate=INF), dict(theta=NAN)):
            with pytest.raises(ValueError, match="finite"):
                topica.TensorLDA(3, **kw)
    finally:
        topica.enable_experimental(was)


def test_valid_hyperparameters_still_construct_and_fit():
    m = LDA(num_topics=2, beta=0.01, alpha_sum=1.0, seed=42)
    m.fit(DOCS, iters=20)
    assert not np.any(np.isnan(m.topic_word))
    assert m.topic_word.shape == (2, len(m.vocabulary))


# --- 2. Degenerate (empty) corpus ------------------------------------------

def test_empty_documents_rejected():
    m = LDA(num_topics=2, seed=42)
    with pytest.raises(ValueError):
        m.fit([[], [], []], iters=20)


def test_all_words_filtered_rejected():
    m = LDA(num_topics=2, seed=42)
    # Every word occurs once; min_doc_freq via a Corpus that prunes everything.
    from topica import Corpus
    with pytest.raises(ValueError):
        Corpus.from_documents([["a"], ["b"], ["c"]], min_doc_freq=5)


# --- 3 & 4. Diagnostic misuse ----------------------------------------------

@pytest.fixture(scope="module")
def model():
    m = LDA(num_topics=2, seed=42)
    m.fit(DOCS, iters=30)
    return m


def test_topic_diversity_rejects_non_integer_topn(model):
    with pytest.raises(ValueError):
        topica.topic_diversity(model.topic_word, model.vocabulary)  # vocab as topn


def test_coherence_rejects_raw_matrix(model):
    with pytest.raises(ValueError):
        topica.coherence(model.topic_word, DOCS)


def test_coherence_rejects_empty_texts(model):
    with pytest.raises(ValueError):
        topica.coherence(model, [])


def test_diagnostics_on_model_still_work(model):
    assert 0.0 < topica.topic_diversity(model) <= 1.0
    assert topica.coherence(model, DOCS).shape == (2,)


# --- 5. frex weight range --------------------------------------------------

@pytest.mark.parametrize("w", [-1.0, 2.0, -0.01, 1.01, NAN])
def test_frex_rejects_bad_weight(model, w):
    with pytest.raises(ValueError):
        topica.frex(model, w=w)


def test_frex_valid_weight_works(model):
    out = topica.frex(model, w=0.5)
    assert len(out) == 2 and isinstance(out[0][0][0], str)


# --- 6. NaN/Inf in covariate and embedding matrices (issue #100) -----------

BIGGER_DOCS = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"]] * 10
N = len(BIGGER_DOCS)


def _nan_matrix(n, cols=2, nan_row=0, nan_col=0):
    x = np.ones((n, cols))
    x[nan_row, nan_col] = float("nan")
    return x


def _inf_matrix(n, cols=2):
    x = np.ones((n, cols))
    x[0, 0] = float("inf")
    return x


def test_stm_prevalence_nan_raises():
    m = topica.STM(num_topics=2, seed=42)
    prev = _nan_matrix(N)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(BIGGER_DOCS, prevalence=prev, iters=5)


def test_stm_prevalence_inf_raises():
    m = topica.STM(num_topics=2, seed=42)
    prev = _inf_matrix(N)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(BIGGER_DOCS, prevalence=prev, iters=5)


def test_dmr_features_nan_raises():
    m = topica.DMR(num_topics=2, seed=42)
    feats = _nan_matrix(N)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(BIGGER_DOCS, feats, iters=5)


def test_keyatm_covariates_nan_raises():
    keywords = {"topic_a": ["a", "b"], "topic_b": ["c", "d"]}
    m = topica.KeyATM(keywords, seed=42)
    covs = _nan_matrix(N)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(BIGGER_DOCS, iters=5, covariates=covs)


def test_embedding_doc_embeddings_nan_raises():
    rng = np.random.default_rng(0)
    docs = BIGGER_DOCS
    emb = rng.standard_normal((N, 8))
    emb[2, 3] = float("nan")
    m = topica.BERTopic(min_cluster_size=5, seed=1)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(docs, emb)


def test_embedding_doc_embeddings_valid_passes():
    rng = np.random.default_rng(0)
    docs = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"]] * 20
    emb = rng.standard_normal((len(docs), 4))
    m = topica.BERTopic(min_cluster_size=5, seed=1)
    # Should not raise; cluster count may be 0 (just warns).
    m.fit(docs, emb)


@pytest.mark.parametrize("ctor", [topica.BERTopic, topica.Top2Vec])
def test_embedding_min_cluster_size_over_corpus_no_panic(ctor):
    # issue #122: min_cluster_size larger than the corpus used to panic inside
    # petal-clustering's MST. It must now resolve to a clean num_topics=0.
    docs = [["a", "b"], ["c", "d"]]
    emb = np.random.default_rng(0).standard_normal((2, 4))
    m = ctor(min_cluster_size=100, seed=1)
    with pytest.warns(UserWarning, match="found no clusters"):
        m.fit(docs, emb)
    assert m.num_topics == 0


# --- 7. iters bounds (issue #103) -----------------------------------------
# iters=0 is a supported "initialize only" operation (e.g. inspecting the seed
# prior before any sweep; see test_seeded_gsdmm_contracts), so it is NOT an
# error. Negative iters is rejected by PyO3's usize conversion (OverflowError).

def test_lda_iters_negative_raises():
    m = topica.LDA(num_topics=2, seed=42)
    with pytest.raises((ValueError, OverflowError)):
        m.fit(DOCS, iters=-1)


def test_lda_iters_zero_is_accepted():
    m = topica.LDA(num_topics=2, seed=42)
    m.fit(DOCS, iters=0)  # initialize without sweeping; must not raise
    assert m.topic_word.shape[0] == 2


def test_lda_iters_positive_works():
    m = topica.LDA(num_topics=2, seed=42)
    m.fit(DOCS, iters=5)
    assert m.topic_word.shape[0] == 2
