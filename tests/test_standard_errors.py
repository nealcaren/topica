"""The general standard-error facility (issue #15): method-of-composition by
default, bootstrap for top-words / embedding models, with an alignment-quality
flag that suppresses SEs where matching is unstable."""

import numpy as np
import pytest

import topica


def _planted(seed=0, n=240):
    """A covariate x that shifts prevalence between two word blocks, with a
    vocabulary large enough that the two topics are genuinely distinct."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    docs = []
    for i in range(n):
        p0 = 1.0 / (1.0 + np.exp(-2.0 * x[i]))
        block = a if rng.random() < p0 else b
        docs.append(list(rng.choice(block, size=12)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=250)
    return m, corpus, x


def test_corpus_doc_lengths_aligns_with_doc_topic():
    m, corpus, _ = _planted()
    assert len(corpus.doc_lengths) == np.asarray(m.doc_topic).shape[0]
    assert all(n > 0 for n in corpus.doc_lengths)


def test_model_family_detection():
    m, _, _ = _planted()
    assert topica.effects.model_family(m) == "dirichlet"
    assert topica.effects.model_family(topica.models.STM(2)) == "logistic_normal"
    assert topica.effects.model_family(topica.models.BERTopic(min_cluster_size=5)) == "none"


# Issue #21: model_family must classify the *whole* registry, not just the
# handful of models that happened to expose `alpha`. `model_family` inspects the
# class (a PyO3 getter is a class attribute even before fit), so unfitted
# instances are enough to pin every model's family. New models added without a
# deliberate family land here as a failure.
_FAMILY_REGISTRY = [
    # collapsed-Gibbs / Dirichlet doc-topic posterior
    ("LDA", lambda: topica.models.LDA(2), "dirichlet"),
    ("HDP", lambda: topica.models.HDP(), "dirichlet"),
    ("KeyATM", lambda: topica.models.KeyATM({"a": ["x"]}, num_topics=2), "dirichlet"),
    ("SeededLDA", lambda: topica.models.SeededLDA({"a": ["x"], "b": ["y"]}), "dirichlet"),
    ("LabeledLDA", lambda: topica.models.LabeledLDA(), "dirichlet"),
    ("SupervisedLDA", lambda: topica.models.SupervisedLDA(num_topics=2), "dirichlet"),
    ("DMR", lambda: topica.models.DMR(2), "dirichlet"),
    ("PA", lambda: topica.models.PA(num_super=2, num_sub=4), "dirichlet"),
    ("PT", lambda: topica.models.PT(num_topics=2, num_pseudo=10), "dirichlet"),
    ("SAGE", lambda: topica.models.SAGE(2), "dirichlet"),
    # logistic-normal (variational eta posterior)
    ("STM", lambda: topica.models.STM(2), "logistic_normal"),
    ("CTM", lambda: topica.models.CTM(2), "logistic_normal"),
    # no theta posterior -> method='bootstrap'. GSDMM is a Dirichlet *mixture*
    # (one topic per document), so a Dirichlet theta draw would misstate its
    # uncertainty; it stays "none" deliberately, alongside the embedding models.
    ("GSDMM", lambda: topica.models.GSDMM(num_topics=5), "none"),
    ("BERTopic", lambda: topica.models.BERTopic(min_cluster_size=5), "none"),
    ("Top2Vec", lambda: topica.models.Top2Vec(), "none"),
    ("ETM", lambda: topica.models.ETM(2), "none"),
    ("ProdLDA", lambda: topica.models.ProdLDA(2), "none"),
    ("FASTopic", lambda: topica.models.FASTopic(2), "none"),
]


@pytest.mark.parametrize("name, make, expected", _FAMILY_REGISTRY,
                         ids=[r[0] for r in _FAMILY_REGISTRY])
def test_model_family_registry(name, make, expected):
    assert topica.effects.model_family(make()) == expected


def _fit_for_composition():
    """Fit one model of each Dirichlet family on a shared toy corpus, with the
    extra inputs each one needs, so composition_theta can be exercised."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(12)]
    docs = [list(rng.choice(vocab, size=10)) for _ in range(40)]
    corpus = topica.Corpus.from_documents(docs)
    y = rng.normal(size=len(docs))
    X = rng.normal(size=(len(docs), 1))

    models = {}
    m = topica.models.LDA(3, seed=1); m.fit(corpus, iters=120); models["LDA"] = m
    # Explicit concentration so HDP instantiates >1 topic on this small random
    # corpus (the default 0.1/0.1 is tuned for real-scale corpora).
    m = topica.models.HDP(seed=1, alpha=1.0, gamma=1.0); m.fit(corpus, iters=120); models["HDP"] = m
    m = topica.models.KeyATM({"a": ["w0"], "b": ["w1"]}, num_topics=3)
    m.fit(corpus, iters=120); models["KeyATM"] = m
    m = topica.models.SeededLDA({"a": ["w0"], "b": ["w1"]}, residual=1)
    m.fit(corpus, iters=120); models["SeededLDA"] = m
    m = topica.models.LabeledLDA(seed=1); m.fit(docs, [["x", "y"]] * len(docs), iters=120)
    models["LabeledLDA"] = m
    m = topica.models.SupervisedLDA(num_topics=3, seed=1); m.fit(docs, y, iters=8)
    models["SupervisedLDA"] = m
    m = topica.models.DMR(num_topics=3, seed=1, optimize_interval=25, burn_in=20)
    m.fit(docs, X, feature_names=["x"], iters=120, num_samples=2, sample_interval=10)
    models["DMR"] = m
    m = topica.models.PA(num_super=2, num_sub=4, seed=1); m.fit(corpus, iters=120)
    models["PA"] = m
    m = topica.models.PT(num_topics=3, num_pseudo=10, seed=1); m.fit(corpus, iters=120)
    models["PT"] = m
    m = topica.models.SAGE(num_topics=3, seed=1, optimize_interval=25, burn_in=20)
    m.fit(docs, ["g"] * len(docs), iters=120, num_samples=2, sample_interval=10)
    models["SAGE"] = m
    return corpus, models


def test_composition_theta_runs_for_every_dirichlet_model():
    """Issue #21: composition_theta must not raise for any Dirichlet model, and
    must return a genuine posterior sample (varying across draws), not a silently
    repeated point estimate that would zero out tempotm's between-draw variance."""
    corpus, models = _fit_for_composition()
    n = len(corpus.doc_lengths)
    for name, model in models.items():
        assert topica.effects.model_family(model) == "dirichlet", name
        k = np.asarray(model.doc_topic).shape[1]
        # HDP's `alpha` is the DP concentration scalar by design; every other
        # Dirichlet model exposes a per-topic prior aligned with doc_topic.
        if name != "HDP":
            assert np.asarray(model.alpha).shape == (k,), name
        draws = topica.effects.composition_theta(model, corpus, nsims=5)
        assert draws.shape == (5, n, k), name
        assert draws.std(axis=0).max() > 0.0, name


def test_composition_effect_inflates_over_ols():
    # The within-document Dirichlet approximation (keep_theta_draws=False) adds
    # 1/N_d sampling noise on top of the point estimate, so method-of-composition
    # SEs are always >= naive OLS that treats theta as fixed. (Real retained
    # draws, issue #31, instead reflect model confidence and can fall below OLS
    # for a well-identified corpus like this one; see
    # test_real_draws_reflect_identifiability in test_theta_draws.py.)
    _, corpus, x = _planted()
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=250, keep_theta_draws=False)
    X = x[:, None]
    ols = topica.effects.estimate_effect(m.doc_topic, X, feature_names=["x"])
    moc = topica.effects.standard_errors(m, corpus, of="effect", X=X, feature_names=["x"], nsims=30)
    # K=2: the two topics' slopes are exact negatives; composition SEs are >= OLS.
    assert np.sign(moc[0].coef[1]) == -np.sign(moc[1].coef[1])
    for t in range(2):
        assert moc[t].se[1] >= ols[t].se[1] - 1e-9


def test_composition_self_sufficient_for_gibbs_and_rejects_embedding():
    # A fitted LDA/keyATM/SeededLDA retains its own per-document lengths (issue
    # #32), so even with draws disabled the Dirichlet fallback needs no corpus=.
    _, corpus, _ = _planted()
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=250, keep_theta_draws=False)
    prev = topica.effects.standard_errors(m, of="prevalence", nsims=10)  # no corpus needed
    assert len(prev) == 2

    # HDP now retains doc_lengths, so it is also self-sufficient (phase 5).
    hdp = topica.models.HDP(seed=1)
    hdp.fit(corpus, iters=120)
    prev_hdp = topica.effects.standard_errors(hdp, of="prevalence", nsims=5)
    assert len(prev_hdp) == hdp.num_topics

    bert = topica.models.BERTopic(min_cluster_size=5)
    with pytest.raises(ValueError, match="bootstrap"):
        topica.effects.standard_errors(bert, corpus, of="prevalence", method="composition", nsims=10)


def test_composition_top_words_is_rejected():
    m, corpus, _ = _planted()
    with pytest.raises(ValueError, match="bootstrap"):
        topica.effects.standard_errors(m, corpus, of="top_words", method="composition")


def test_topic_correlation_ci_well_shaped_and_coherent():
    # CTM/STM topic-correlation credible interval from the logistic-normal posterior:
    # K x K, symmetric, unit diagonal, ci_low <= estimate <= ci_high in every cell,
    # and the estimate is the posterior mean of the draw correlations.
    _, corpus, _ = _planted()
    m = topica.models.CTM(num_topics=3, seed=1)
    m.fit(corpus, iters=40)
    res = topica.effects.topic_correlation_ci(m, nsims=150, ci=0.9, seed=0)
    K = m.num_topics
    for arr in (res.estimate, res.se, res.ci_low, res.ci_high):
        assert arr.shape == (K, K)
    assert np.allclose(res.estimate, res.estimate.T)
    assert np.allclose(np.diag(res.estimate), 1.0)
    assert np.all(res.ci_low <= res.estimate + 1e-9)
    assert np.all(res.estimate <= res.ci_high + 1e-9)
    assert np.all(res.ci_low <= res.ci_high + 1e-9)
    assert np.all(np.isfinite(res.se))
    # Wider interval mass gives a wider band.
    narrow = topica.effects.topic_correlation_ci(m, nsims=150, ci=0.5, seed=0)
    span90 = float(np.sum(res.ci_high - res.ci_low))
    span50 = float(np.sum(narrow.ci_high - narrow.ci_low))
    assert span90 > span50


def test_topic_correlation_ci_rejects_non_logistic_normal():
    _, corpus, _ = _planted()
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=50)
    with pytest.raises(TypeError, match="logistic-normal"):
        topica.effects.topic_correlation_ci(m)


def test_bootstrap_prevalence_matches_composition_on_clean_data():
    m, corpus, _ = _planted()
    comp = topica.effects.standard_errors(m, corpus, of="prevalence", nsims=40)
    boot = topica.effects.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                  n_boot=40, topn=5, iters=150, seed=0)
    for t in range(2):
        assert boot[t].reliable
        assert boot[t].alignment_quality > 0.5 and boot[t].alignment_margin > 0.1
        # Same ballpark (within 3x); both are honest, small SEs here.
        assert 0.2 < boot[t].se / comp[t].se < 5.0


def test_bootstrap_top_words_inclusion_probs():
    m, corpus, _ = _planted()
    res = topica.effects.standard_errors(m, corpus, of="top_words", method="bootstrap",
                                 n_boot=30, topn=5, iters=150, seed=0)
    assert len(res) == 2
    for r in res:
        assert r.reliable
        # Each top word has an inclusion probability in [0, 1]; stable topics keep
        # their words most of the time.
        probs = [p for (_, p, _, _) in r.words]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert max(probs) > 0.5


def test_alignment_margin_flags_indistinct_topics():
    # topn larger than the distinct words per topic makes every topic's top-word
    # set identical: the Jaccard looks perfect but the match is arbitrary. The
    # margin diagnostic must catch this and suppress the SE.
    m, corpus, _ = _planted()
    res = topica.effects.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                 n_boot=20, topn=20, iters=150, seed=0)
    for r in res:
        assert r.alignment_margin < 0.1
        assert not r.reliable
        assert np.isnan(r.se)


def test_bootstrap_via_refit_hook():
    # The general hook: a user-supplied refit(doc_indices) -> fitted model, the
    # path embedding models use (resample embeddings alongside documents).
    m, corpus, _ = _planted()
    docs = corpus.documents()

    def refit(picks):
        mm = topica.models.LDA(2, seed=int(picks[0]) + 1)
        mm.fit([docs[i] for i in picks], iters=150)
        return mm

    res = topica.effects.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                 n_boot=20, topn=5, refit=refit, seed=0)
    assert len(res) == 2 and all(r.reliable for r in res)
