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


def _supervised_corpus(n_docs=80, seed=0):
    """Two topics whose response means are far apart, so a fit predicts y well."""
    rng = np.random.default_rng(seed)
    ta = ["cat", "dog", "pet", "fur", "paw", "tail"]
    tb = ["bank", "loan", "money", "rate", "cash", "debt"]
    docs, y = [], []
    for i in range(n_docs):
        if i % 2 == 0:
            docs.append(list(rng.choice(ta, size=rng.integers(8, 16))))
            y.append(rng.normal(2.0, 0.5))
        else:
            docs.append(list(rng.choice(tb, size=rng.integers(8, 16))))
            y.append(rng.normal(8.0, 0.5))
    return docs, np.array(y)


def test_supervised_oof_basic():
    docs, y = _supervised_corpus()
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        seed=13, fit_kwargs={"iters": 20},
    )
    assert result.kind == "supervised"
    assert result.oof_predictions.shape == (len(docs),)
    assert result.scored_mask.all()  # kfold, no drops on this dense corpus
    # Topics separate the two response regimes -> strong OOF R2.
    assert result.aggregate["r2_pooled"] > 0.5
    assert 0.0 <= result.aggregate["coverage_90"] <= 1.0
    assert "R2" in result.summary()


def test_supervised_deterministic():
    docs, y = _supervised_corpus()
    make = lambda: topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        seed=13, fit_kwargs={"iters": 20},
    )
    a, b = make(), make()
    np.testing.assert_array_equal(
        np.nan_to_num(a.oof_predictions), np.nan_to_num(b.oof_predictions)
    )


def test_supervised_empty_doc_not_polluting_metrics():
    """The Gate A blocker: SupervisedLDA.predict returns (0.0, sigma2) for an empty/
    all-OOV doc. Predicting on the transformed corpus must DROP those docs, not score
    a fabricated 0.0 against a response near 8."""
    docs, y = _supervised_corpus(n_docs=80)
    # Make several docs empty; they must be excluded from the OOF metrics, not scored 0.
    for i in (1, 3, 5, 7):
        docs[i] = []
    with pytest.warns(UserWarning, match="dropped from out-of-fold"):
        result = topica.cross_validate(
            lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
            seed=13, fit_kwargs={"iters": 20},
        )
    # The empty docs are NaN in the OOF vector (never a fabricated 0.0).
    for i in (1, 3, 5, 7):
        assert np.isnan(result.oof_predictions[i])
    assert not result.scored_mask[[1, 3, 5, 7]].any()
    # And the pooled RMSE is still sane (a 0.0-vs-8 leak would blow it up).
    assert result.aggregate["rmse_pooled"] < 2.0


def test_supervised_y_length_assert():
    docs, y = _supervised_corpus(n_docs=80)
    with pytest.raises(ValueError, match="expected n_docs"):
        topica.cross_validate(
            lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y[:70], folds=4
        )


def test_supervised_y_nonfinite_rejected():
    docs, y = _supervised_corpus(n_docs=80)
    y = y.copy()
    y[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        topica.cross_validate(
            lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4
        )


def test_supervised_needs_predict():
    docs, y = _supervised_corpus(n_docs=60)
    with pytest.raises(ValueError, match="no predict"):
        topica.cross_validate(
            lambda s: topica.LDA(2, seed=s), docs, y=y, folds=4, fit_kwargs={"iters": 20}
        )


def test_supervised_n_jobs_bit_identical_to_serial():
    docs, y = _supervised_corpus(n_docs=80)
    serial = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        seed=13, fit_kwargs={"iters": 20}, n_jobs=1,
    )
    parallel = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        seed=13, fit_kwargs={"iters": 20}, n_jobs=3,
    )
    # Thread pool must be bit-identical to serial, not merely close.
    np.testing.assert_array_equal(
        np.nan_to_num(serial.oof_predictions), np.nan_to_num(parallel.oof_predictions)
    )
    np.testing.assert_array_equal(
        np.nan_to_num(serial.oof_std), np.nan_to_num(parallel.oof_std)
    )
    assert serial.aggregate["rmse_pooled"] == parallel.aggregate["rmse_pooled"]


def test_supervised_constant_y_fold_keeps_rmse_mae():
    """A constant-response fold has a valid RMSE/MAE (only R2 is undefined); those
    folds must stay in the macro population (Gate B)."""
    from topica.crossval import _supervised_aggregate

    # Two folds' worth of per-fold records: one normal, one constant-y (r2 NaN).
    per_fold = [
        {"fold": 0, "rmse": 1.0, "mae": 0.8, "r2": 0.5},
        {"fold": 1, "rmse": 2.0, "mae": 1.6, "r2": float("nan")},
    ]
    y = np.array([1.0, 2.0, 3.0, 4.0])
    oof = np.array([1.1, 2.1, 2.9, 4.2])
    std = np.full(4, 0.5)
    mask = np.ones(4, bool)
    agg, _ = _supervised_aggregate(y, oof, std, mask, per_fold, (0.9,))
    # Both folds contribute to macro RMSE/MAE; only R2 drops the constant fold.
    assert agg["rmse_macro"]["n_valid_folds"] == 2
    assert agg["mae_macro"]["n_valid_folds"] == 2
    assert agg["r2_macro"]["n_valid_folds"] == 1


def test_supervised_calibration_well_calibrated():
    """On a clean linear-response synthetic, calibration slope ~ 1, intercept ~ 0."""
    docs, y = _supervised_corpus(n_docs=120)
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=5,
        seed=13, fit_kwargs={"iters": 25},
    )
    assert 0.7 < result.aggregate["calibration_slope"] < 1.3
    assert abs(result.aggregate["calibration_intercept"]) < 2.0
    assert result.calibration_table is not None
    assert set(result.calibration_table.columns) >= {"bin", "n", "mean_pred", "mean_obs"}


def test_supervised_manifest_records_settings_and_replayable():
    docs, y = _supervised_corpus(n_docs=60)
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        fit_kwargs={"iters": 20},
    )
    cv = result.manifest.cv
    assert cv["model"]["settings"] is not None       # full constructor settings
    assert cv["replayable"] is True                  # settings captured, no callback
    assert "coverage_z" in cv["supervised"]          # exact Gaussian quantiles
    assert "calibration_rule" in cv["supervised"]


def test_supervised_manifest_records_response():
    docs, y = _supervised_corpus(n_docs=60)
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        fit_kwargs={"iters": 20},
    )
    cv = result.manifest.cv
    assert cv["path"] == "supervised"
    assert cv["supervised"]["response_shape"] == [60]
    assert "response_hash" in cv["supervised"]
    assert cv["supervised"]["coverage_levels"] == [0.90, 0.95]


def test_supervised_temporal_oof_mask_excluded():
    docs, y = _supervised_corpus(n_docs=80)
    times = np.arange(80)
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y,
        strategy="temporal", times=times, folds=5, fit_kwargs={"iters": 20},
    )
    # Initial-window docs are never tested -> NaN OOF, excluded from scored_mask.
    assert not result.scored_mask.all()
    assert np.isnan(result.oof_predictions[0])


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


# ---------------------------------------------------------------------------
# Covariate-effect fold-stability (#705) — keyATM covariate lambda
# ---------------------------------------------------------------------------


def test_covariate_effect_stability_keyatm():
    """keyATM covariate lambda gets a sign-agreement + magnitude-correlation surface,
    NOT predictive coverage, and never lands in a coverage_ field."""
    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    with pytest.warns(UserWarning):  # the marginal-covariate warning still fires
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs,
            covariates={"covariates": X},
            folds=3,
            fit_kwargs={"iters": 300},
        )
    cs = result.covariate_stability
    assert cs is not None
    assert -1.0 <= cs["sign_agreement"] <= 1.0
    assert cs["n_pairs"] == 3  # 3 fold pairs from 3 folds
    assert "per_feature" in cs and len(cs["per_feature"]) == cs["n_features"]
    # It is explicitly NOT coverage and must not leak into any coverage_ field.
    assert "not predictive coverage" in cs["note"].lower()
    assert not any(k.startswith("coverage_") for k in result.aggregate)
    assert not any(k.startswith("coverage") for k in cs)
    assert "covariate-effect stability" in result.summary()
    assert "NOT predictive coverage" in result.summary()


def test_covariate_effect_stability_none_without_covariates():
    """Plain LDA has no lambda, so there is nothing to report."""
    docs, _ = _synthetic_corpus(n_docs=40)
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=3, fit_kwargs={"iters": 30}
    )
    assert result.covariate_stability is None


def test_covariate_effect_stability_self_pair_agrees():
    """Identical covariate fits must have perfect sign-agreement and magnitude corr."""
    from topica.crossval import _covariate_effect_stability

    docs, labels = _synthetic_corpus(n_docs=60)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    m = topica.KeyATM(keywords, num_topics=2, seed=1)
    m.fit(topica.Corpus.from_documents(docs), covariates=X, iters=300)
    cs = _covariate_effect_stability([m, m])
    assert cs is not None
    assert cs["sign_agreement"] == 1.0
    # A self-pair is a perfect line; corr is 1.0 (or NaN only if lambda is constant).
    assert np.isnan(cs["magnitude_correlation"]) or cs["magnitude_correlation"] > 0.99


def test_covariate_effect_stability_headline_is_feature_macro():
    """The headline must be the mean of the per-feature stats, not a pooled corr over
    raw cells (which a high-variance covariate would dominate — adversarial finding)."""
    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    # Two covariates on wildly different scales; the fix must weight them equally.
    X = np.column_stack([labels.astype(float), 1000.0 * labels.astype(float)])
    with pytest.warns(UserWarning):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs, covariates={"covariates": X}, folds=3, fit_kwargs={"iters": 300},
        )
    cs = result.covariate_stability
    per = cs["per_feature"]
    assert len(per) == 2
    sa = [v["sign_agreement"] for v in per.values() if np.isfinite(v["sign_agreement"])]
    mc = [v["magnitude_correlation"] for v in per.values()
          if np.isfinite(v["magnitude_correlation"])]
    if sa:
        assert cs["sign_agreement"] == pytest.approx(float(np.mean(sa)))
    if mc:
        assert cs["magnitude_correlation"] == pytest.approx(float(np.mean(mc)))


def test_covariate_effect_stability_near_zero_is_undefined():
    """An under-trained keyATM learns all-zero effects; the diagnostic must report
    UNDEFINED, not a spurious sign-agreement of 1.0 (the fresh-user trap, #705)."""
    from topica.crossval import _covariate_effect_stability

    docs, labels = _synthetic_corpus(n_docs=60)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    # A tiny number of iterations leaves keyATM's feature_effects at exactly 0.
    models = []
    for s in (1, 2):
        m = topica.KeyATM(keywords, num_topics=2, seed=s)
        m.fit(topica.Corpus.from_documents(docs), covariates=X, iters=5)
        models.append(m)
    if not all(np.all(m.feature_effects[:, 1:] == 0) for m in models):
        pytest.skip("keyATM warmed up faster than expected; not the near-zero regime")
    cs = _covariate_effect_stability(models)
    assert cs["effects_near_zero"] is True
    assert np.isnan(cs["sign_agreement"])  # NOT 1.0
    assert "WARNING" in cs["note"] and "more fit iterations" in cs["note"]


def test_covariate_effect_stability_names_and_labels():
    """covariate_names= flows into the per-feature table and wins over placeholders."""
    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = np.column_stack([labels.astype(float), (1 - labels).astype(float)])
    with pytest.warns(UserWarning):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs, covariates={"covariates": X}, covariate_names=["party", "not_party"],
            folds=3, fit_kwargs={"iters": 300},
        )
    assert list(result.covariate_stability["per_feature"].keys()) == ["party", "not_party"]


def test_covariate_names_length_mismatch_errors():
    from topica.crossval import _resolve_covariate_names

    with pytest.raises(ValueError, match="must match|learned"):
        _resolve_covariate_names(["a", "b", "c"], {"covariates": None}, 2)


def test_covariate_tuple_gives_clear_error():
    """Passing the (matrix, names) tuple from one_hot straight through is caught with
    an actionable message, not a raw numpy error (fresh-user docs trap, #705)."""
    docs, labels = _synthetic_corpus(n_docs=40)
    tup = (labels.reshape(-1, 1).astype(float), ["x"])
    with pytest.raises(ValueError, match="unpack one_hot|is a tuple"):
        topica.cross_validate(
            lambda s: topica.STM(2, seed=s), docs,
            covariates={"prevalence": tup}, folds=3, fit_kwargs={"iters": 20},
        )


def test_covariate_effect_stability_reports_topics_compared():
    """topics_compared / n_topics / partial_alignment expose how many of K topics each
    fold pair actually compared (so a silent topic-drop can't inflate stability)."""
    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    with pytest.warns(UserWarning):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=3, seed=s),
            docs, covariates={"covariates": X}, folds=3, fit_kwargs={"iters": 300},
        )
    cs = result.covariate_stability
    assert cs["n_topics"] == 3
    tc = cs["topics_compared"]
    assert 1 <= tc["min"] <= tc["max"] <= 3
    assert cs["partial_alignment"] == bool(tc["min"] < 3)


def test_covariate_stability_frame_and_labels():
    """covariate_stability_frame() is a per-feature DataFrame; a bare covariate matrix
    gets a kwarg-based label, never a placeholder feature_0 with >1 covariate."""
    pd = pytest.importorskip("pandas")
    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = np.column_stack([labels.astype(float), (1 - labels).astype(float)])
    with pytest.warns(UserWarning):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs, covariates={"covariates": X}, folds=3, fit_kwargs={"iters": 300},
        )
    frame = result.covariate_stability_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns[:2]) == ["feature", "sign_agreement"]
    # Names fall back to the covariate kwarg, indexed — not positional feature_0.
    assert list(frame["feature"]) == ["covariates[0]", "covariates[1]"]
    # No-covariate run returns None from the frame accessor.
    plain = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=3, fit_kwargs={"iters": 20}
    )
    assert plain.covariate_stability_frame() is None


# ---------------------------------------------------------------------------
# plot_cv (#705) — the viz panel
# ---------------------------------------------------------------------------


def test_plot_cv_topic_path():
    pytest.importorskip("matplotlib")
    import topica.viz as viz

    docs, _ = _synthetic_corpus()
    result = topica.cross_validate(
        lambda s: topica.LDA(2, seed=s), docs, folds=4, fit_kwargs={"iters": 30}
    )
    panel = viz.plot_cv(result)
    df = panel.to_frame()
    assert len(df) == 4
    fig = panel.to_png()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_cv_supervised_path():
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import topica.viz as viz

    docs, y = _supervised_corpus(n_docs=80)
    result = topica.cross_validate(
        lambda s: topica.SupervisedLDA(2, seed=s), docs, y=y, folds=4,
        fit_kwargs={"iters": 25},
    )
    panel = viz.plot_cv(result)
    assert panel._kind == "supervised"
    fig = panel.to_png()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_cv_covariate_panel():
    """When the run carries covariate-effect stability, the topic figure adds the
    covariate band (the reason a covariate model was run)."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    import topica.viz as viz

    docs, labels = _synthetic_corpus(n_docs=90)
    keywords = {"animals": ["cat", "dog", "pet"], "finance": ["bank", "loan", "money"]}
    X = labels.reshape(-1, 1).astype(float)
    with pytest.warns(UserWarning):
        result = topica.cross_validate(
            lambda s: topica.KeyATM(keywords, num_topics=2, seed=s),
            docs, covariates={"covariates": X}, folds=3, fit_kwargs={"iters": 300},
        )
    panel = viz.plot_cv(result)
    assert panel._has_covariate() is True
    fig = panel.to_png()
    assert fig is not None
    plt.close(fig)
