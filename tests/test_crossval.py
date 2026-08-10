"""Orchestrator tests for cross_validate (#701, PR1) — topic-model path."""

import numpy as np
import pytest

import topica


def _synthetic_corpus(n_docs=60, seed=0):
    """Two clearly separated topics so a fit is meaningful on small folds."""
    rng = np.random.default_rng(seed)
    topic_a = ["cat", "dog", "pet", "fur", "paw", "tail"]
    topic_b = ["bank", "loan", "money", "rate", "cash", "debt"]
    docs = []
    labels = []
    for i in range(n_docs):
        vocab = topic_a if i % 2 == 0 else topic_b
        docs.append(list(rng.choice(vocab, size=rng.integers(8, 16))))
        labels.append(i % 2)
    return docs, np.array(labels)


def test_cross_validate_lda_basic():
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s),
        docs,
        folds=5,
        seed=13,
        fit_kwargs={"iters": 50},
    )
    assert len(result.per_fold) == 5
    assert "perplexity" in result.aggregate
    for rec in result.per_fold:
        assert rec["perplexity"] > 0
        assert rec["n_eval_tokens"] > 0
        assert rec["covariate_conditioned"] is False  # LDA has no covariates
        assert rec["vocab_size"] > 0


def test_cross_validate_deterministic():
    docs, _ = _synthetic_corpus()
    make = lambda: topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, seed=13, fit_kwargs={"iters": 30}
    )
    a, b = make(), make()
    pa = [r["perplexity"] for r in a.per_fold]
    pb = [r["perplexity"] for r in b.per_fold]
    assert pa == pb


def test_cross_validate_per_fold_vocab_default():
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=5, fit_kwargs={"iters": 30}
    )
    assert result.vocab == "per_fold"
    # Per-fold perplexity present; aggregate is macro-only.
    assert result.aggregate["perplexity"]["mean"] > 0
    assert "per-fold" in result.summary()


def test_cross_validate_fixed_vocab_warns():
    docs, _ = _synthetic_corpus()
    with pytest.warns(UserWarning, match="feature selection"):
        result = topica.cross_validate(
            lambda s: topica.LDA(2, seed=s),
            docs,
            folds=4,
            vocab="fixed",
            fit_kwargs={"iters": 30},
        )
    assert result.vocab == "fixed"


def test_cross_validate_stm_prevalence_conditioned():
    docs, labels = _synthetic_corpus()
    X = labels.reshape(-1, 1).astype(float)
    result = topica.cross_validate(
        lambda s: topica.STM(2, seed=s),
        docs,
        covariates={"prevalence": X},
        folds=4,
        seed=13,
        fit_kwargs={"iters": 40},
    )
    for rec in result.per_fold:
        assert rec["covariate_conditioned"] is True  # STM conditioned on prevalence


def test_stm_covariate_status_in_summary():
    docs, labels = _synthetic_corpus()
    X = labels.reshape(-1, 1).astype(float)
    result = topica.cross_validate(
        lambda s: topica.STM(2, seed=s),
        docs,
        covariates={"prevalence": X},
        folds=4,
        fit_kwargs={"iters": 40},
    )
    assert "conditioned held-out inference" in result.summary()


def test_marginal_covariate_warns_and_labels():
    """keyATM cannot condition held-out scoring on covariates; the marginal fallback
    must warn AND say so in summary() (the Tier-1 sample-user trap)."""
    docs, labels = _synthetic_corpus()
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    with pytest.warns(UserWarning, match="MARGINAL|did not condition|cannot condition"):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs,
            covariates={"covariates": X},
            folds=3,
            fit_kwargs={"iters": 30},
        )
    assert "MARGINAL" in result.summary()
    assert all(r["covariate_conditioned"] is False for r in result.per_fold)


def test_covariate_length_mismatch_errors():
    docs, labels = _synthetic_corpus(n_docs=60)
    with pytest.raises(ValueError, match="expected n_docs"):
        topica.cross_validate(
            lambda s: topica.STM(2, seed=s),
            docs,
            covariates={"prevalence": labels[:50].reshape(-1, 1).astype(float)},
            folds=4,
        )


def test_bare_dataframe_covariate_rejected():
    pd = pytest.importorskip("pandas")
    docs, labels = _synthetic_corpus()
    with pytest.raises(ValueError, match="dict keyed"):
        topica.cross_validate(
            lambda s: topica.STM(2, seed=s),
            docs,
            covariates=pd.DataFrame({"x": labels}),
            folds=4,
        )


def test_covariate_fit_kwargs_collision():
    docs, labels = _synthetic_corpus()
    X = labels.reshape(-1, 1).astype(float)
    with pytest.raises(ValueError, match="collides"):
        topica.cross_validate(
            lambda s: topica.STM(2, seed=s),
            docs,
            covariates={"prevalence": X},
            fit_kwargs={"prevalence": X},
            folds=4,
        )


def test_supplied_folds_leak_rejected():
    """A hand-built Folds with an overlapping split must be re-validated, not trusted."""
    from topica.crossval import Folds

    docs, _ = _synthetic_corpus(n_docs=20)
    bad = Folds(
        splits=[
            (np.array([0, 1, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64)),  # 1 in both
            (np.array([3, 4], dtype=np.int64), np.array([0, 2], dtype=np.int64)),
        ],
        strategy="kfold",
        seed=13,
        fold_seeds=[1, 2],
        oof_mask=np.ones(20, dtype=bool),
        n_docs=20,
    )
    with pytest.raises(ValueError, match="both train and test"):
        topica.cross_validate(lambda s: topica.LDA(2, seed=s), docs, folds=bad)


def test_supplied_folds_seed_count_mismatch_rejected():
    from topica.crossval import make_folds
    from dataclasses import replace

    docs, _ = _synthetic_corpus(n_docs=20)
    f = make_folds(20, folds=4, seed=13)
    broken = replace(f, fold_seeds=f.fold_seeds[:2])  # too few seeds
    with pytest.raises(ValueError, match="fold seeds"):
        topica.cross_validate(lambda s: topica.LDA(2, seed=s), docs, folds=broken)


def test_unknown_covariate_key_hard_fails():
    docs, labels = _synthetic_corpus()
    X = labels.reshape(-1, 1).astype(float)
    with pytest.raises(ValueError, match="does not accept covariate"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s),  # LDA has no covariates
            docs,
            covariates={"prevalence": X},
            folds=4,
        )


def test_stm_covariates_alias_conditions_scoring():
    """A covariate passed under the 'covariates' alias must still condition held-out
    inference (not just the fit) — the alias is canonicalized for both."""
    docs, labels = _synthetic_corpus()
    X = labels.reshape(-1, 1).astype(float)
    result = topica.cross_validate(
        lambda s: topica.STM(2, seed=s),
        docs,
        covariates={"covariates": X},  # alias for prevalence
        folds=4,
        fit_kwargs={"iters": 40},
    )
    assert all(r["covariate_conditioned"] is True for r in result.per_fold)


def test_score_fn_without_fit_fn_rejected():
    docs, _ = _synthetic_corpus()
    with pytest.raises(ValueError, match="score_fn requires"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s),
            docs,
            score_fn=lambda m, td, ti, s: {},
            folds=4,
        )


def test_metrics_param_not_yet_supported():
    docs, _ = _synthetic_corpus()
    with pytest.raises(NotImplementedError, match="metric selection"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s), docs, metrics=["perplexity"], folds=4
        )


def test_nan_times_rejected():
    docs, _ = _synthetic_corpus(n_docs=30)
    times = np.arange(30, dtype=float)
    times[5] = np.nan
    with pytest.raises(ValueError, match="missing .*timestamp"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s), docs, strategy="temporal", times=times, folds=4
        )


def test_prebuilt_corpus_per_fold_warns():
    docs, _ = _synthetic_corpus()
    corpus = topica.Corpus.from_documents(docs, min_doc_freq=2)
    with pytest.warns(UserWarning, match="pre-built Corpus"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s), corpus, folds=4, fit_kwargs={"iters": 30}
        )


def test_manifest_records_content_and_model():
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, fit_kwargs={"iters": 30}
    )
    cv = result.manifest.cv
    assert "doc_content_hash" in cv
    assert cv["model"]["class"] == "LDA"
    assert cv["metric_params"]["topn"] == 10


def test_supervised_y_not_yet_supported():
    docs, labels = _synthetic_corpus()
    with pytest.raises(NotImplementedError, match="PR2"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s), docs, y=labels, folds=4
        )


def test_fit_fn_requires_score_fn():
    docs, _ = _synthetic_corpus()
    with pytest.raises(ValueError, match="score_fn"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s),
            docs,
            fit_fn=lambda td, ti, s: None,
            folds=4,
        )


def test_fit_fn_score_fn_escape_hatch():
    docs, _ = _synthetic_corpus()

    def fit_fn(train_docs, train_idx, seed_fold):
        m = topica.LDA(2, seed=seed_fold)
        m.fit(topica.Corpus.from_documents(train_docs), iters=30)
        return m

    def score_fn(model, test_docs, test_idx, seed_fold):
        return {"perplexity": topica.perplexity(model, test_docs, seed=seed_fold)}

    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, fit_fn=fit_fn, score_fn=score_fn, folds=4
    )
    assert all(r["perplexity"] > 0 for r in result.per_fold)


def test_manifest_records_cv():
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, fit_kwargs={"iters": 30}
    )
    cv = getattr(result.manifest, "cv", None)
    assert cv is not None
    assert cv["strategy"] == "kfold"
    assert len(cv["splits"]) == 4
    assert len(cv["fold_seeds"]) == 4


def test_temporal_oof_mask_excludes_initial_window():
    docs, _ = _synthetic_corpus()
    times = np.arange(len(docs))
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s),
        docs,
        strategy="temporal",
        times=times,
        folds=5,
        fit_kwargs={"iters": 30},
    )
    assert not result.folds.oof_mask.all()  # initial window never tested


def test_grouped_end_to_end():
    docs, _ = _synthetic_corpus()
    groups = np.repeat(np.arange(12), 5)
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s),
        docs,
        strategy="grouped",
        groups=groups,
        folds=4,
        fit_kwargs={"iters": 30},
    )
    assert len(result.per_fold) == 4


def test_short_doc_covariate_codrop():
    """A test doc with <2 tokens is skipped in scoring; its covariate row must be
    co-dropped so STM's prevalence transform stays aligned (Gate A-B3)."""
    docs, labels = _synthetic_corpus(n_docs=60)
    docs[1] = ["cat"]  # a 1-token doc that scoring will skip
    docs[3] = []  # an empty doc
    X = labels.reshape(-1, 1).astype(float)
    result = topica.cross_validate(
        lambda s: topica.STM(2, seed=s),
        docs,
        covariates={"prevalence": X},
        folds=4,
        seed=13,
        fit_kwargs={"iters": 30},
    )
    # No crash and every fold scored some docs.
    assert all(r["n_scored_docs"] > 0 for r in result.per_fold)


def test_cross_fold_stability_present():
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, fit_kwargs={"iters": 50}
    )
    assert result.stability is not None
    assert 0.0 <= result.stability["mean"] <= 1.0
    # Direction check: cleanly-separated two-topic folds must be SIMILAR across
    # folds, not distant. (A 1 - cosine inversion would report ~0.1 here.)
    assert result.stability["mean"] > 0.5
    assert result.stability["n_pairs"] == 6  # C(4, 2)
    assert "stability" in result.summary()


def test_cross_fold_stability_is_cosine_not_distance():
    """A self-pair must score ~1.0 (identical topics), not ~0.0 (the inverted bug)."""
    from topica.crossval import _cross_fold_stability

    docs, _ = _synthetic_corpus(n_docs=40)
    m = topica.LDA(2, seed=1)
    m.fit(topica.Corpus.from_documents(docs), iters=50)
    stab = _cross_fold_stability([m, m])
    assert stab["mean"] > 0.99  # identical fits are maximally stable


def test_to_frame():
    pytest.importorskip("pandas")
    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, fit_kwargs={"iters": 30}
    )
    df = result.to_frame()
    assert len(df) == 4
    assert "perplexity" in df.columns
