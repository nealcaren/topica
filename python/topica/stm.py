"""STM-style analysis toolkit on top of topica's Gibbs topic models.

These are post-hoc analyses of a fitted model's outputs (the topic-word matrix
``topic_word`` = φ and the document-topic matrix ``doc_topic`` = θ), mirroring
the user-facing functions of the R ``stm`` package:

- :func:`estimate_effect` — regress topic proportions on document covariates
  (≈ ``stm::estimateEffect``).
- :func:`label_topics` / :func:`frex` — prob / FREX / lift / score topic words
  (≈ ``stm::labelTopics``).
- :func:`topic_correlation` — topic-correlation network (≈ ``stm::topicCorr``).
- :func:`find_thoughts` — representative documents per topic
  (≈ ``stm::findThoughts``).
- :func:`search_k` — fit across topic counts and report quality
  (≈ ``stm::searchK``).

Everything operates on numpy arrays, so it works with any model here (LDA, DMR,
LabeledLDA). :func:`estimate_effect` does ordinary OLS on a point estimate of θ,
or — given posterior draws from :func:`posterior_theta_samples` (an STM/CTM
variational posterior) — the **method of composition**, pooling per-draw
regressions by Rubin's rules so the standard errors propagate topic-estimation
uncertainty, following the same method-of-composition procedure as R ``stm``'s
``estimateEffect``. The θ draws use R ``estimateEffect``'s default **Global**
uncertainty (one shared topic covariance across documents) unless you pass
``uncertainty="local"`` (each document's own variational covariance) or
``"none"``. Nonlinear and interaction terms are built with :func:`spline` and
:func:`interaction`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# estimateEffect: covariate -> topic-proportion regression
# ---------------------------------------------------------------------------

@dataclass
class TopicEffect:
    """OLS of one topic's proportion on the covariates.

    ``coef``/``se``/``z``/``ci_low``/``ci_high``/``pvalue`` are aligned to
    ``feature_names`` **positionally**. With ``add_intercept=True`` (the
    :func:`~topica.effects.estimate_effect` default) ``feature_names[0]`` is ``"intercept"``,
    so ``coef[0]`` is the baseline constant, *not* your covariate — a common way to
    accidentally report the wrong number. Prefer name-keyed access
    (:meth:`effect_of`, :attr:`by_feature`, :meth:`to_frame`) over positional
    indexing.
    """

    topic: int
    feature_names: list[str]
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    r_squared: float
    vcov: np.ndarray = None  # full (p, p) coefficient covariance (Rubin-pooled)
    varcomp: dict = None  # random-effect variance components (sd), when random= is set
    reliable: bool = True
    message: str = ""

    @property
    def pvalue(self) -> np.ndarray:
        """Two-sided normal p-value per feature, from the Wald statistic ``z``.

        ``P(|Z| > |z|) = erfc(|z| / sqrt(2))``; ``nan`` only where ``z`` is ``nan``
        (e.g. an unreliable bootstrap SE). An infinite ``z`` (a nonzero coefficient
        with a zero SE) gives ``0.0``, consistent with the formula. Aligned to
        :attr:`feature_names`."""
        import math

        z = np.atleast_1d(np.asarray(self.z, dtype=float))
        out = np.full(z.shape, np.nan)
        valid = ~np.isnan(z)  # inf is valid -> erfc(inf)=0.0; only nan stays nan
        out[valid] = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z[valid]]
        return out.reshape(np.shape(self.z))

    def effect_of(self, feature: str) -> dict:
        """Named-feature accessor: the row for ``feature`` as a dict of ``coef``,
        ``se``, ``z``, ``ci_low``, ``ci_high``, ``pvalue``. Raises ``KeyError`` with
        the available names if ``feature`` is not a covariate, or ``ValueError`` if
        the name is not unique (ambiguous) — the safe alternative to guessing a
        positional index (see the class note on the intercept)."""
        names = list(self.feature_names)
        if feature not in names:
            raise KeyError(
                f"{feature!r} is not a covariate of this effect; available: {names}"
            )
        if names.count(feature) > 1:
            raise ValueError(
                f"{feature!r} appears {names.count(feature)} times in feature_names, "
                "so named access is ambiguous; read the row positionally via "
                ".coef/.se/... or .to_frame() instead."
            )
        j = names.index(feature)
        coef = np.atleast_1d(self.coef)
        se = np.atleast_1d(self.se)
        ci_low = np.atleast_1d(self.ci_low)
        ci_high = np.atleast_1d(self.ci_high)
        pval = np.atleast_1d(self.pvalue)
        return {
            "coef": float(coef[j]),
            "se": float(se[j]),
            "z": float(np.atleast_1d(self.z)[j]),
            "ci_low": float(ci_low[j]),
            "ci_high": float(ci_high[j]),
            "pvalue": float(pval[j]),
        }

    @property
    def by_feature(self) -> dict:
        """``{feature_name: effect_of(name)}`` — every covariate keyed by name, so a
        result reads without positional indexing."""
        return {name: self.effect_of(name) for name in self.feature_names}

    def as_dict(self) -> dict:
        pval = self.pvalue
        return {
            "topic": self.topic,
            **{
                f"{name}": {
                    "coef": float(self.coef[j]),
                    "se": float(self.se[j]),
                    "z": float(self.z[j]),
                    "pvalue": float(pval[j]),
                    "ci": (float(self.ci_low[j]), float(self.ci_high[j])),
                }
                for j, name in enumerate(self.feature_names)
            },
            "r_squared": self.r_squared,
        }

    def to_frame(self):
        """Return a tidy pandas DataFrame, one row per feature.

        Columns are ``topic``, ``feature``, ``coef``, ``se``, ``z``, ``pvalue``,
        ``ci_low``, ``ci_high``, and ``r_squared`` (the topic's value, repeated).
        The named ``feature`` column is the safe way to read a specific covariate's
        effect — an ``intercept`` row is present when ``add_intercept=True``. Because
        the ``topic`` column is included, concatenating the frames from a whole
        :func:`estimate_effect` call gives one row per (topic, feature)::

            import pandas as pd
            effects = topica.effects.estimate_effect(model, X, feature_names=names)
            table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "topic": self.topic,
                "feature": list(self.feature_names),
                "coef": np.asarray(self.coef, dtype=float),
                "se": np.asarray(self.se, dtype=float),
                "z": np.asarray(self.z, dtype=float),
                "pvalue": np.asarray(self.pvalue, dtype=float),
                "ci_low": np.asarray(self.ci_low, dtype=float),
                "ci_high": np.asarray(self.ci_high, dtype=float),
                "r_squared": self.r_squared,
            }
        )


class EffectList(list):
    """The result of :func:`estimate_effect`: a ``list`` of per-topic
    :class:`TopicEffect` objects, one per topic.

    It *is* a plain ``list`` (index it, iterate it, ``len()`` it), with one
    convenience the siblings :func:`search_k` / :func:`topic_table` /
    :func:`bootstrap_stability` also give: a container-level :meth:`to_frame`, so
    ``estimate_effect(...).to_frame()`` works without the
    ``pd.concat([e.to_frame() for e in effects])`` boilerplate.

    Use :meth:`to_frame`, not ``pandas.DataFrame(effects)``: the two differ here.
    :meth:`to_frame` gives the tidy long table (one row per topic-feature, named
    columns); ``pandas.DataFrame(effects)`` would build a wide frame of raw
    :class:`TopicEffect` fields (array-valued cells, internal attributes). Unlike
    :func:`topic_table`, whose result hands straight to ``pandas.DataFrame``, an
    ``EffectList`` does not.
    """

    def to_frame(self):
        """One tidy pandas DataFrame for the whole call, one row per
        (topic, feature).

        Concatenates every :meth:`TopicEffect.to_frame`, so the columns are
        ``topic``, ``feature``, ``coef``, ``se``, ``z``, ``pvalue``, ``ci_low``,
        ``ci_high``, ``r_squared`` (see :meth:`TopicEffect.to_frame`). Returns an
        empty DataFrame when there are no effects.
        """
        import pandas as pd

        if not self:
            return pd.DataFrame()
        return pd.concat([e.to_frame() for e in self], ignore_index=True)


def _coerce_design(X, feature_names):
    """Coerce a design-matrix argument to a float64 ``(n, p)`` array.

    Accepts a numpy array (any numeric dtype), a numeric pandas/Polars
    DataFrame or Series, or a list of number rows — so callers no longer have to
    pre-cast with ``.to_numpy(float)``. A 1-D input becomes an ``(n, 1)`` column.
    When ``feature_names`` is not given and the input is a DataFrame, the column
    labels are used as feature names. A non-numeric column (strings, a pandas
    ``Categorical``) raises a directive error pointing at
    :func:`topica.design.design_matrix` / :func:`topica.design.one_hot` rather than surfacing a
    cryptic numpy cast failure.

    Returns ``(X_float64, names_or_None)``.
    """
    inferred = None
    if hasattr(X, "select_dtypes"):  # pandas DataFrame
        bad = [str(c) for c in X.select_dtypes(exclude="number").columns]
        if bad:
            raise ValueError(
                f"covariate column(s) {bad} are non-numeric and cannot be cast to "
                "float. Encode categorical covariates first with "
                "topica.design.design_matrix(formula, data) or topica.design.one_hot(...), then "
                "pass the resulting numeric matrix."
            )
        inferred = [str(c) for c in X.columns]
    elif hasattr(X, "name") and hasattr(X, "to_numpy") and getattr(X, "ndim", None) == 1:
        # pandas/Polars Series: a single named covariate column.
        inferred = [str(X.name)] if X.name is not None else None

    try:
        arr = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(
            "could not convert the covariates to a float64 matrix; encode "
            "categorical covariates with topica.design.design_matrix / topica.design.one_hot "
            f"first (numpy: {e})"
        ) from e
    if arr.ndim == 1:
        arr = arr[:, None]
    names = list(feature_names) if feature_names is not None else inferred
    return arr, names


def _ols(y, X, hat, XtX_inv, dof, weights=None):
    """One (weighted) OLS fit. Returns (beta, cov, r2).

    With ``weights`` (a length-n vector), this is weighted least squares: ``hat``
    and ``XtX_inv`` are expected to already carry the weights (the caller builds
    them once), and the residual sum of squares, total sum of squares, and R^2 are
    weighted. The classical covariance is ``(rss/dof) · (X'WX)^-1`` — matching R
    ``lm`` / faSTM's ``estimateEffect`` weighted path.
    """
    beta = hat @ y
    resid = y - X @ beta
    if weights is None:
        rss = float(resid @ resid)
        sigma2 = rss / dof
        tss = float(((y - y.mean()) ** 2).sum())
    else:
        rss = float((weights * resid * resid).sum())
        sigma2 = rss / dof
        ybar = float((weights * y).sum() / weights.sum())
        tss = float((weights * (y - ybar) ** 2).sum())
    cov = sigma2 * XtX_inv
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return beta, cov, r2


def _link_inv(eta, link):
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    return eta


def _sandwich(X, bread, score_resid, groups, n, p, weights=None, *, hc1=True):
    """Robust covariance ``bread · meat · bread``. With `groups` (a list of index
    arrays) the cluster-robust CR1 estimator; otherwise heteroskedasticity-robust.
    `score_resid` is the estimating-equation residual (y−μ). With `weights`
    (survey weights), the score rows become ``w_i · X_i · resid_i`` and `bread` is
    the weighted ``(X'WX)^-1`` the caller supplies — matching faSTM's weighted
    cluster-robust meat.

    `hc1` applies the ``n/(n-p)`` finite-sample factor to the heteroskedasticity-
    robust (non-cluster) meat: use it for the linear model (matches statsmodels
    HC1), and turn it off for a GLM, whose robust SE convention is HC0 — statsmodels
    treats HC0/HC1 identically for a GLM, and the Papke–Wooldridge fractional-logit
    robust SE is HC0 (no df factor). The cluster CR1 factor is unaffected."""
    sr = score_resid if weights is None else weights * score_resid
    if groups is None:
        meat = X.T @ (X * (sr ** 2)[:, None])
        cov = bread @ meat @ bread
        if hc1:
            cov *= n / max(n - p, 1)                   # HC1 small-sample factor
    else:
        g_count = len(groups)
        meat = np.zeros((p, p))
        for g in groups:
            s = X[g].T @ sr[g]
            meat += np.outer(s, s)
        cov = bread @ meat @ bread
        if g_count > 1:                                # CR1 small-sample factor
            cov *= (g_count / (g_count - 1)) * ((n - 1) / max(n - p, 1))
    return cov


def _glm_irls(y, X, link, *, iters=50, tol=1e-9, sw=None):
    """Iteratively reweighted least squares for a quasi-likelihood GLM (binomial
    for ``logit``, Poisson for ``log``). With ``sw`` (survey weights) the IRLS
    weights are scaled by ``sw`` (weighted estimating equations). Returns (beta,
    final combined weights)."""
    p = X.shape[1]
    beta = np.zeros(p)
    W = np.ones(X.shape[0])
    for _ in range(iters):
        eta = X @ beta
        mu = _link_inv(eta, link)
        if link == "logit":
            mu = np.clip(mu, 1e-8, 1 - 1e-8)
            gprime = 1.0 / (mu * (1.0 - mu))
            W = mu * (1.0 - mu)                        # 1 / (g'(μ)² · V(μ))
        else:  # log / quasi-Poisson
            mu = np.clip(mu, 1e-8, None)
            gprime = 1.0 / mu
            W = mu
        if sw is not None:
            W = W * sw
        z = eta + (y - mu) * gprime                    # working response
        new = np.linalg.pinv(X.T @ (X * W[:, None])) @ (X.T @ (W * z))
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta, np.clip(W, 1e-12, None)


def _fit_one(y, X, *, link, groups, hat, XtX_inv, dof, weights=None):
    """Fit one topic's regression. ``link`` is identity (OLS) / logit (fractional
    logit) / log (quasi-Poisson); ``groups`` (or None) selects cluster-robust vs
    classical/robust covariance; ``weights`` (or None) are survey weights (WLS).
    When weighted, ``hat`` and ``XtX_inv`` already carry the weights. Returns
    (beta, cov, r2)."""
    n, p = X.shape
    if link == "identity" and groups is None:
        return _ols(y, X, hat, XtX_inv, dof, weights=weights)  # classical (W)LS
    if link == "identity":
        beta = hat @ y
        cov = _sandwich(X, XtX_inv, y - X @ beta, groups, n, p, weights=weights)
        mu = X @ beta
    else:
        beta, W = _glm_irls(y, X, link, sw=weights)
        bread = np.linalg.pinv(X.T @ (X * W[:, None]))
        mu = _link_inv(X @ beta, link)
        # GLM robust SE is HC0 (no n/(n-p) factor): matches statsmodels' GLM
        # sandwich and the Papke–Wooldridge fractional-logit convention (#533).
        cov = _sandwich(X, bread, y - mu, groups, n, p, weights=weights, hc1=False)  # ψ_i = w_i X_i(y_i−μ_i)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - mu) ** 2).sum()) / tss if tss > 0 else 0.0
    return beta, cov, r2


def _parse_random_intercept(random):
    """Parse an lme4-style random-effect spec into a grouping-factor name.

    Accepts ``"(1 | group)"`` or ``"1 | group"``. Only a random *intercept*
    (``1 | group``) is supported; a random slope (``x | group``) raises. Returns
    the bare group-factor column name.
    """
    spec = random.strip()
    if spec.startswith("(") and spec.endswith(")"):
        spec = spec[1:-1].strip()
    if "|" not in spec:
        raise ValueError(
            "random= must be an lme4-style bar, e.g. '(1 | group)'"
        )
    lhs, group = (s.strip() for s in spec.split("|", 1))
    if lhs not in ("1", ""):
        raise NotImplementedError(
            "only random intercepts '(1 | group)' are supported; random slopes "
            f"like '({lhs} | {group})' are not yet implemented"
        )
    if "|" in group or not group:
        raise ValueError(
            "random= supports a single grouping factor, e.g. '(1 | group)'"
        )
    return group


def _golden_min(f, lo, hi, tol=1e-6, max_iter=200):
    """Minimize a unimodal ``f`` on ``[lo, hi]`` by golden-section search.
    Returns the minimizing x. Dependency-light (no scipy)."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0
    a, b = lo, hi
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if h < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            h = b - a
            c = a + invphi2 * h
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            h = b - a
            d = a + invphi * h
            fd = f(d)
    return (a + b) / 2.0


def _reml_random_intercept(y, X, groups):
    """REML fit of ``y = Xβ + b_{group} + ε`` with a single random intercept.

    ``groups`` is a length-n integer array of grouping-factor codes. Matches
    ``lme4::lmer(y ~ X + (1 | group), REML=TRUE)`` for the fixed effects: the
    variance ratio ``λ = σ²_group / σ²_resid`` is chosen by minimizing the
    profiled REML deviance (a smooth 1-D problem, so golden-section search finds
    the same optimum lme4 does), and the fixed-effect covariance is
    ``σ̂²_resid · (XᵀV⁻¹X)⁻¹``. The random-intercept covariance ``V`` is
    block-diagonal by group, so every GLS quantity has a closed form per group
    (Sherman-Morrison), no n×n solve.

    Returns ``dict(beta, vcov, var_group, var_resid)``. ``r²`` is left to the
    caller (undefined for a mixed model, reported as NaN like faSTM/lme4).
    """
    n, p = X.shape
    dof = max(n - p, 1)
    XtX = X.T @ X
    Xty = X.T @ y
    # Per-group sufficient statistics: column sums of X, sum of y, and size.
    codes = np.asarray(groups)
    uniq = np.unique(codes)
    gidx = [np.where(codes == g)[0] for g in uniq]
    gstats = [(X[idx].sum(axis=0), float(y[idx].sum()), len(idx)) for idx in gidx]

    def _fit_at(lam):
        A = XtX.copy()
        b = Xty.copy()
        log_m = 0.0
        for sx, sy, nj in gstats:
            c = lam / (1.0 + lam * nj)
            A -= c * np.outer(sx, sx)
            b -= c * sx * sy
            log_m += np.log1p(lam * nj)
        a_inv = np.linalg.pinv(A)
        beta = a_inv @ b
        resid = y - X @ beta
        r_m_r = 0.0
        for idx, (_, _, nj) in zip(gidx, gstats):
            rj = resid[idx]
            c = lam / (1.0 + lam * nj)
            s = float(rj.sum())
            r_m_r += float(rj @ rj) - c * s * s
        sigma2 = max(r_m_r / dof, 1e-300)
        _, logdet_a = np.linalg.slogdet(A)
        deviance = dof * np.log(sigma2) + log_m + logdet_a
        return deviance, beta, a_inv, sigma2

    def _dev(t):
        return _fit_at(np.exp(t))[0]

    # Search log λ over a wide bracket, then compare against the λ→0 (OLS)
    # boundary — the singular fit lme4 allows when the group variance is ~0.
    t_lo, t_hi = np.log(1e-8), np.log(1e8)
    t_star = _golden_min(_dev, t_lo, t_hi)
    dev_star, beta, a_inv, sigma2 = _fit_at(np.exp(t_star))
    lam = float(np.exp(t_star))
    dev0, beta0, a_inv0, sigma20 = _fit_at(0.0)
    if dev0 <= dev_star:
        lam, beta, a_inv, sigma2 = 0.0, beta0, a_inv0, sigma20
    return {
        "beta": beta,
        "vcov": sigma2 * a_inv,
        "var_group": lam * sigma2,
        "var_resid": sigma2,
    }


def _pooled_random(theta, X, group_codes, topic_list):
    """Per-topic random-intercept coefficients, Rubin-pooled across θ draws.

    Mirrors :func:`_pooled_coefficients` (returns ``(beta, Sigma, r2=nan)`` per
    topic) but fits ``_reml_random_intercept`` per draw, so the method-of-
    composition standard errors carry both the mixed-model sampling error (the
    within term) and the topic-estimation uncertainty (the between term).
    """
    pooled = theta.ndim == 3
    p = X.shape[1]
    out = []
    for t in topic_list:
        if pooled:
            m = theta.shape[0]
            betas = np.empty((m, p))
            within = np.zeros((p, p))
            for i in range(m):
                fit = _reml_random_intercept(theta[i, :, t], X, group_codes)
                betas[i] = fit["beta"]
                within += fit["vcov"]
            within /= m
            beta = betas.mean(axis=0)
            between = np.cov(betas, rowvar=False) if m > 1 else np.zeros((p, p))
            Sigma = within + (1.0 + 1.0 / m) * np.atleast_2d(between)
            out.append((beta, Sigma, float("nan")))
        else:
            fit = _reml_random_intercept(theta[:, t], X, group_codes)
            out.append((fit["beta"], fit["vcov"], float("nan")))
    return out


def _pooled_coefficients(theta, X, *, link, groups, hat, XtX_inv, dof, topic_list,
                         weights=None):
    """Fit per-topic regressions and pool by Rubin's rules.

    Returns a list of ``(beta, Sigma, r2)`` tuples — one per topic in
    ``topic_list`` — where ``Sigma`` is the full ``(p, p)`` posterior covariance
    (not just the diagonal). Both :func:`estimate_effect` and
    :func:`predicted_prevalence` call this so their coefficient posteriors never
    diverge.

    Parameters
    ----------
    theta : ndarray
        Either ``(n, K)`` for a point estimate or ``(M, n, K)`` for draws.
    X : ndarray
        Design matrix ``(n, p)`` — intercept already prepended.
    link, groups, hat, XtX_inv, dof : as in :func:`estimate_effect`.
    topic_list : list[int]
        Topic indices to fit (validated by the caller).

    Returns
    -------
    list of ``(beta (p,), Sigma (p, p), r2 float)``
    """
    pooled = theta.ndim == 3
    if pooled:
        nsims_inner = theta.shape[0]
    # The fast batched path is plain (unweighted) OLS without clustering; survey
    # weights route through the per-topic slow path (weighted hat/XtX_inv supplied).
    fast = link == "identity" and groups is None and weights is None
    p = X.shape[1]
    n = X.shape[0]

    # Fast batched path for plain OLS without clustering.
    if fast and pooled:
        Yt = theta[:, :, topic_list]                          # (M, n, T)
        B = np.einsum("pn,snt->spt", hat, Yt)                 # (M, p, T)
        R = Yt - np.einsum("np,spt->snt", X, B)
        ss = np.einsum("snt,snt->st", R, R)                   # (M, T)
        within_scale = (ss / dof).mean(axis=0)                # (T,)
        beta_mean = B.mean(axis=0)                            # (p, T)
        tss = ((Yt - Yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)  # (M, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_all = np.where(tss > 0, 1.0 - ss / tss, 0.0).mean(axis=0)  # (T,)
        out = []
        for i in range(len(topic_list)):
            between = np.cov(B[:, :, i], rowvar=False) if nsims_inner > 1 else np.zeros((p, p))
            Sigma = within_scale[i] * XtX_inv + (1.0 + 1.0 / nsims_inner) * np.atleast_2d(between)
            out.append((beta_mean[:, i], Sigma, float(r2_all[i])))
        return out

    if fast:
        Y = theta[:, topic_list]                              # (n, T)
        B = hat @ Y                                           # (p, T)
        R = Y - X @ B
        ss = np.einsum("nt,nt->t", R, R)                      # (T,)
        tss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)         # (T,)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_all = np.where(tss > 0, 1.0 - ss / tss, 0.0)
        out = []
        for i in range(len(topic_list)):
            sigma2 = float(ss[i]) / dof
            Sigma = sigma2 * XtX_inv
            out.append((B[:, i], Sigma, float(r2_all[i])))
        return out

    # Slow path: GLM / cluster-robust, per-topic.
    out = []
    for t in topic_list:
        if pooled:
            betas = np.empty((nsims_inner, p))
            within = np.zeros((p, p))
            r2s = np.empty(nsims_inner)
            for m in range(nsims_inner):
                b, cov_m, r2_m = _fit_one(theta[m, :, t], X, link=link, groups=groups,
                                           hat=hat, XtX_inv=XtX_inv, dof=dof,
                                           weights=weights)
                betas[m] = b
                within += cov_m
                r2s[m] = r2_m
            within /= nsims_inner
            beta = betas.mean(axis=0)
            between = np.cov(betas, rowvar=False) if nsims_inner > 1 else np.zeros((p, p))
            Sigma = within + (1.0 + 1.0 / nsims_inner) * np.atleast_2d(between)
            out.append((beta, Sigma, float(r2s.mean())))
        else:
            beta, cov, r2 = _fit_one(theta[:, t], X, link=link, groups=groups,
                                     hat=hat, XtX_inv=XtX_inv, dof=dof,
                                     weights=weights)
            out.append((beta, cov, r2))
    return out


def estimate_effect(
    doc_topic,
    X=None,
    *,
    data=None,
    formula=None,
    feature_names=None,
    topics=None,
    add_intercept=True,
    ci=0.95,
    cluster=None,
    weights=None,
    random=None,
    link="identity",
    corpus=None,
    nsims=None,
    seed=0,
    uncertainty="global",
):
    """Regress each topic's proportion on document covariates.

    Pass a point estimate of θ for an ordinary OLS, or a *stack of posterior
    draws* of θ for the **method of composition** — the uncertainty-propagating
    procedure R ``stm`` uses (Treier & Jackman 2008). With draws, each one is
    regressed and the results are pooled by Rubin's rules, so the reported
    standard errors include the topic-estimation uncertainty, not just OLS
    sampling error. Get draws with :func:`posterior_theta_samples`.

    A **point** θ gives OLS standard errors that treat the topic proportions as
    fixed and so *understate* uncertainty. For a model with a θ posterior, prefer
    draws (or pass the model with ``nsims=``). For a cluster/embedding model with no
    posterior (e.g. BERTopic), method-of-composition is unavailable; pass the model
    (not just its ``doc_topic`` array) so this is flagged, and use
    ``standard_errors(..., method="bootstrap")`` to quantify uncertainty.

    ``uncertainty`` (STM/CTM only, when the θ posterior is drawn here via a model
    + ``nsims``) selects the draw covariance, matching R ``stm``'s
    ``thetaPosterior`` ``type=``. It defaults to ``"global"`` — R
    ``estimateEffect``'s default — which draws every document from one shared
    covariance (the global topic-model uncertainty) and so widens the intervals
    relative to ``"local"`` (each document's own variational covariance, topica's
    former behavior). ``"none"`` propagates no topic uncertainty (OLS on the point
    θ). It has no effect when you pass a precomputed draw array or a Dirichlet
    (Gibbs) model.

    For paper-grade inference two extras matter:

    - ``cluster`` — a length-``num_docs`` array of group labels (e.g. speaker,
      user, outlet). Text data is almost always nested, and ignoring it
      understates uncertainty. Supplying it switches the standard errors to the
      **cluster-robust** (CR1) sandwich estimator. (With posterior draws, each
      draw is clustered and the per-draw covariances are then Rubin-pooled.)
    - ``link`` — ``"identity"`` (default OLS), ``"logit"`` (fractional logit, via
      binomial quasi-likelihood), or ``"log"`` (quasi-Poisson). Because topic
      proportions live in ``[0, 1]``, the logit link keeps fitted values in
      bounds where OLS can wander outside them (Papke & Wooldridge). Non-identity
      links report heteroskedasticity- or cluster-robust standard errors; the
      heteroskedasticity-robust GLM SE is HC0 (no ``n/(n-p)`` factor), matching
      statsmodels' GLM sandwich and the Papke–Wooldridge convention. The identity
      link without ``cluster`` reports classical OLS standard errors; under
      ``cluster`` both links use the CR1 cluster-robust sandwich.
    - ``weights`` — a length-``num_docs`` array of (survey) weights, or a column
      name in ``data``. Switches to weighted least squares: documents enter the
      regression in proportion to their weight, so a weighted sample (e.g. a
      survey-weighted corpus, or documents weighted by length) estimates the
      population-level effect. Composes with ``cluster`` (weighted cluster-robust
      SEs) and with ``link``. Matches faSTM's weighted ``estimateEffect``.
    - ``random`` — an lme4-style random-intercept term ``"(1 | group)"`` (with
      ``group`` a column of ``data``). Fits a mixed model — the fixed-effect design
      plus a random intercept per group — by REML for each posterior draw, then
      Rubin-pools the fixed effects, matching faSTM's ``estimateEffect(... ~ x +
      (1 | group))``. Use it when documents are nested in units (state, outlet,
      author) whose baseline topic level varies: the random intercept soaks up that
      between-unit variation so the fixed-effect SEs are not understated. The
      estimated group and residual standard deviations are attached as
      ``TopicEffect.varcomp``. Only a random *intercept* is supported (not random
      slopes), with ``link="identity"`` and no ``cluster``/``weights``.

    Specifying the design. Give the covariates one of two ways: a prebuilt design
    matrix as ``X`` (with ``feature_names``), or an R-style ``formula`` together
    with a ``data`` frame, which builds ``X`` for you via
    :func:`topica.design.design_matrix`. **Use the same design you fit the model with.**
    The effects regression is on the covariates you pass here, not on whatever
    went into ``STM.fit``; if they differ, the coefficients answer a different
    question than the model. The reliable pattern is to build the design once and
    pass the identical ``X`` (or the identical ``formula`` + ``data``) to both
    ``fit`` and ``estimate_effect``.

    Parameters
    ----------
    doc_topic : array or fitted model
        Either ``(num_docs, num_topics)`` — a point θ (``model.doc_topic``) for
        plain OLS — or ``(nsims, num_docs, num_topics)`` — posterior θ draws for
        method-of-composition pooling. You may also pass the **fitted model**
        itself: with ``nsims`` (and ``corpus=`` for a Gibbs model) the right θ
        posterior is drawn for you; without ``nsims`` its point θ is used.
    X : array (num_docs, p)
        Document covariates (design matrix); build nonlinear/interaction terms
        with :func:`spline` / :func:`interaction`. An intercept is prepended when
        ``add_intercept`` is True.
    feature_names : list[str], optional
        Column names for ``X``. Defaults to ``feature_0 ...``.
    data : pandas.DataFrame, optional
        Used with ``formula`` to build the design matrix; ignored when ``X`` is
        given. A string ``cluster`` is read as a column of this frame.
    formula : str, optional
        R-style formula (e.g. ``"~ party + spline(year, df=3)"``) evaluated
        against ``data`` to build ``X`` and ``feature_names``, via
        :func:`topica.design.design_matrix` (needs the optional ``topica[formula]``
        extra). Pass either ``X`` or ``formula`` + ``data``, not both.
    topics : sequence[int], optional
        Restrict to these topics. Defaults to all.
    ci : float
        Confidence level for the (normal-approximation) intervals.
    uncertainty : {"global", "local", "none"}
        Draw covariance for the STM/CTM θ posterior when it is sampled here
        (model + ``nsims``); R ``stm``'s ``thetaPosterior`` ``type=``. Defaults
        to ``"global"`` (R ``estimateEffect``'s default; shared covariance,
        wider CIs). See the discussion above. Ignored for precomputed draws or
        Dirichlet models.

    Returns
    -------
    EffectList
        A ``list`` of one :class:`TopicEffect` per topic (index and iterate it
        like any list). For a tidy long table with one row per (topic, feature),
        call the container's :meth:`~EffectList.to_frame`::

            table = topica.effects.estimate_effect(model, X, feature_names=names).to_frame()

        (equivalent to
        ``pd.concat([e.to_frame() for e in result], ignore_index=True)``).
    """
    # Formula path: build X and feature_names from an R-style formula + a
    # DataFrame. A string `cluster` is read as a column of that frame.
    if formula is not None:
        if data is None:
            raise ValueError("formula= requires data= (a pandas DataFrame).")
        from .formulas import design_matrix

        X, feature_names = design_matrix(formula, data)
        if isinstance(cluster, str):
            cluster = np.asarray(data[cluster])
        if isinstance(weights, str):
            weights = np.asarray(data[weights], dtype=np.float64)
    elif X is None:
        raise ValueError("provide X (a design matrix), or formula= with data=.")
    if isinstance(weights, str):
        raise ValueError("weights= as a column name requires formula= with data=.")

    # Random-intercept spec: read the grouping factor from `data`.
    group_raw = None
    if random is not None:
        if link != "identity":
            raise ValueError("random= is only supported with link='identity'")
        if cluster is not None or weights is not None:
            raise ValueError("random= cannot be combined with cluster= or weights=")
        group_name = _parse_random_intercept(random)
        if data is None or group_name not in getattr(data, "columns", []):
            raise ValueError(
                f"random= needs data= with a '{group_name}' column"
            )
        group_raw = np.asarray(data[group_name])

    # Accept a fitted model as the first argument and draw theta internally: with
    # nsims, the family-appropriate posterior is sampled for method-of-composition
    # standard errors (no hand-wiring a sampler); without it, the point theta is
    # used for plain OLS.
    if uncertainty not in ("global", "local", "none"):
        raise ValueError(
            f"uncertainty must be 'global', 'local', or 'none' (got {uncertainty!r})"
        )
    if hasattr(doc_topic, "doc_topic") and not isinstance(doc_topic, np.ndarray):
        from .effects import composition_theta, model_family

        _model = doc_topic
        if nsims:
            doc_topic = composition_theta(
                _model, corpus, nsims=nsims, seed=seed, uncertainty=uncertainty
            )
        else:
            # A model with no posterior over theta (a cluster/embedding model such as
            # BERTopic) has no method-of-composition path, so this is plain OLS on a
            # point estimate: the topic proportions are treated as fixed and the SEs
            # understate uncertainty. standard_errors()/effect_plot refuse in this
            # case; warn here too rather than returning confident-looking CIs. (A
            # posterior model used at its point theta is the documented OLS baseline
            # and can upgrade via nsims=, so it is not warned.)
            if model_family(_model) == "none":
                warnings.warn(
                    f"{type(_model).__name__} has no posterior over topic "
                    "proportions, so these standard errors are ordinary least "
                    "squares on a point estimate: they treat the proportions as "
                    "fixed and understate uncertainty. Method-of-composition "
                    "intervals are unavailable for this model; use "
                    "standard_errors(model, corpus, of='effect', method='bootstrap') "
                    "to quantify uncertainty.",
                    stacklevel=2,
                )
            doc_topic = np.asarray(_model.doc_topic, dtype=np.float64)

    theta = np.asarray(doc_topic, dtype=np.float64)
    pooled = theta.ndim == 3
    if pooled:
        nsims, n, num_topics = theta.shape
    elif theta.ndim == 2:
        n, num_topics = theta.shape
    else:
        raise ValueError("doc_topic must be 2-D (num_docs, K) or 3-D (nsims, num_docs, K)")

    X, names = _coerce_design(X, feature_names)
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but doc_topic has {n} documents")

    if names is None:
        names = [f"feature_{i}" for i in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError("feature_names length must match X columns")
    if add_intercept:
        X = np.hstack([np.ones((n, 1)), X])
        names = ["intercept"] + names

    if link not in ("identity", "logit", "log"):
        raise ValueError("link must be 'identity', 'logit', or 'log'")
    groups = None
    if cluster is not None:
        cluster = np.asarray(cluster)
        if cluster.shape[0] != n:
            raise ValueError("cluster must have one label per document")
        groups = [np.where(cluster == g)[0] for g in np.unique(cluster)]

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError("weights must have one value per document")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite and non-negative")

    p = X.shape[1]
    # A rank-deficient design (e.g. a bring-your-own full dummy set plus the
    # default add_intercept=True, or an X that already carries its own intercept)
    # is silently solved by the pseudoinverse, which returns an arbitrary
    # minimum-norm split of the collinear effect across the tied columns. The
    # per-column coefficients are then not identified and easy to misread as
    # meaningful. R's estimateEffect warns and adds a ridge; we warn.
    if n >= p:
        rank = np.linalg.matrix_rank(X)
        if rank < p:
            warnings.warn(
                f"design matrix is rank-deficient ({rank} independent columns for "
                f"{p} coefficients): the columns are collinear (a common cause is a "
                f"full set of dummies plus add_intercept=True, or an X that already "
                f"contains an intercept column). The pseudoinverse returns an "
                f"arbitrary minimum-norm split of the shared effect, so the "
                f"individual coefficients are not identified. Drop a redundant "
                f"column (e.g. one_hot(..., drop_first=True) or add_intercept=False), "
                f"or use the formula= path, which handles this for you.",
                stacklevel=2,
            )
    # Cluster-robust SEs need many clusters. With G < 2 the sandwich meat
    # collapses to ~0 (SEs spuriously near zero); with G <= p the cluster vcov is
    # rank-deficient so some coefficients' SEs are understated. Previously silent.
    if groups is not None:
        n_g = len(groups)
        if n_g < 2:
            warnings.warn(
                f"cluster-robust standard errors need at least 2 clusters; got {n_g}. "
                f"The sandwich meat collapses and the reported SEs will be near zero "
                f"(spuriously confident). Drop cluster= or use a coarser grouping.",
                stacklevel=2,
            )
        elif n_g <= p:
            warnings.warn(
                f"cluster-robust standard errors have only {n_g} clusters for {p} "
                f"coefficients: the cluster covariance is rank-deficient (G <= p), so "
                f"some coefficients' SEs are unreliable (understated). Cluster-robust "
                f"inference is trustworthy only with many clusters.",
                stacklevel=2,
            )
    if weights is None:
        XtX_inv = np.linalg.pinv(X.T @ X)
        hat = XtX_inv @ X.T  # (p, n)
    else:
        Xw = X * weights[:, None]
        XtX_inv = np.linalg.pinv(Xw.T @ X)     # (X'WX)^-1
        hat = XtX_inv @ Xw.T                    # weighted: hat @ y = (X'WX)^-1 X'W y
    dof = max(n - p, 1)
    z_crit = _normal_ppf(0.5 + ci / 2.0)  # normal-approx critical value (no scipy)

    topic_list = list(range(num_topics)) if topics is None else list(topics)
    for t in topic_list:
        if t < 0 or t >= num_topics:
            raise ValueError(f"topic {t} out of range (num_topics={num_topics})")

    varcomp_by_topic = None
    if group_raw is not None:
        if group_raw.shape[0] != n:
            raise ValueError("random= group must have one label per document")
        _, group_codes = np.unique(group_raw, return_inverse=True)
        group_name = _parse_random_intercept(random)
        pooled_results = _pooled_random(theta, X, group_codes, topic_list)
        # Variance components: one REML fit per topic on the posterior-mean theta
        # (faSTM reports VarCorr from a single stable refit, not the pooled draws).
        theta_bar = theta.mean(axis=0) if theta.ndim == 3 else theta
        varcomp_by_topic = {}
        for t in topic_list:
            fit = _reml_random_intercept(theta_bar[:, t], X, group_codes)
            varcomp_by_topic[t] = {
                group_name: float(np.sqrt(max(fit["var_group"], 0.0))),
                "residual": float(np.sqrt(max(fit["var_resid"], 0.0))),
            }
    else:
        pooled_results = _pooled_coefficients(
            theta, X, link=link, groups=groups, hat=hat, XtX_inv=XtX_inv, dof=dof,
            topic_list=topic_list, weights=weights,
        )

    out = EffectList()
    for (beta, Sigma, r2), t in zip(pooled_results, topic_list):
        se = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            zvals = np.where(se > 0, beta / se, 0.0)
        out.append(
            TopicEffect(
                topic=t,
                feature_names=names,
                coef=beta,
                se=se,
                z=zvals,
                ci_low=beta - z_crit * se,
                ci_high=beta + z_crit * se,
                r_squared=r2,
                vcov=Sigma,
                varcomp=None if varcomp_by_topic is None else varcomp_by_topic[t],
            )
        )
    return out


@dataclass
class MarginalEffect:
    """One average marginal effect: a (topic, covariate term) pair.

    Produced by :func:`average_marginal_effects`. ``topic_name`` is the topic's
    label. ``ame`` is the average expected change in the topic's proportion for the
    covariate term (a unit change for a continuous covariate, a level-vs-reference
    contrast for a factor), averaged over the observed documents. ``se`` is its
    standard error and ``ci_low``/``ci_high`` the bounds of the confidence
    interval.
    """

    topic: int
    topic_name: str
    term: str
    ame: float
    se: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "topic_name": self.topic_name,
            "term": self.term,
            "ame": self.ame,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass
class AverageMarginalEffects:
    """The full set of average marginal effects for one covariate.

    Returned by :func:`average_marginal_effects`. Iterate ``.effects`` for the
    per-(topic, term) :class:`MarginalEffect` rows, or call :meth:`to_frame` for a
    tidy DataFrame.
    """

    covariate: str
    effects: list

    def __iter__(self):
        return iter(self.effects)

    def __len__(self):
        return len(self.effects)

    def to_frame(self):
        """Return a tidy pandas DataFrame, one row per (topic, term)."""
        import pandas as pd

        return pd.DataFrame([e.as_dict() for e in self.effects])


def average_marginal_effects(
    doc_topic,
    covariate,
    *,
    formula,
    data,
    topics=None,
    h=None,
    ci=0.95,
    cluster=None,
    weights=None,
    corpus=None,
    nsims=None,
    seed=0,
    add_intercept=True,
):
    """Average marginal effects of a covariate on topic prevalence.

    The average expected change in a topic's proportion per unit of ``covariate``,
    averaged over the observed documents. For a **continuous** covariate this is
    the average numeric derivative (central difference); for a **factor** it is the
    average contrast of each non-reference level against the reference level. This
    is cleaner than reading raw regression coefficients, especially when the design
    has splines or interactions, where no single coefficient is the effect (cf. the
    ``margins`` package, and faSTM's ``ame()``).

    The marginal effect is computed on the identity (proportion) scale: each
    topic's prevalence is regressed on the design via the method of composition
    (the same path as :func:`estimate_effect`), and the averaged design-change
    vector is contracted with the per-topic coefficient posterior, propagating
    topic-estimation uncertainty into the standard error via the Rubin-pooled
    coefficient covariance.

    Parameters
    ----------
    doc_topic : array or fitted model
        As in :func:`estimate_effect` — a fitted model (theta drawn internally
        when ``nsims`` is given), a ``(num_docs, K)`` point theta, or
        ``(nsims, num_docs, K)`` posterior draws.
    covariate : str
        Column in ``data`` to compute marginal effects for.
    formula : str
        R-style formula for the design (must reference ``covariate``). Splines are
        replayed with the training knots, so a perturbed covariate uses the same
        basis as the fit.
    data : pandas.DataFrame
        One row per document; the design is rebuilt on perturbed copies of it.
    topics : sequence[int], optional
        Restrict to these topics. Defaults to all.
    h : float, optional
        Step for the numeric derivative of a continuous covariate. Defaults to
        ``0.01 * sd(covariate)``.
    ci : float
        Confidence level for the (normal-approximation) intervals.
    cluster, weights : optional
        Passed through to :func:`estimate_effect` for the underlying regression
        (cluster-robust SEs, survey weights). When ``weights`` is given the
        design-change is averaged with those weights (a population marginal
        effect); otherwise it is a plain sample average.
    corpus, nsims, seed, add_intercept : optional
        As in :func:`estimate_effect`.

    Returns
    -------
    AverageMarginalEffects
        Iterable of :class:`MarginalEffect`; ``.to_frame()`` for a tidy table.
    """
    if formula is None or data is None:
        raise ValueError("average_marginal_effects requires formula= and data=.")
    if covariate not in data.columns:
        raise ValueError(f"covariate {covariate!r} is not a column of data.")

    from .formulas import _KnotCapturingContext, design_matrix, design_matrix_predict

    # Underlying per-topic regression (identity link): gives coef + full vcov.
    effects = estimate_effect(
        doc_topic, formula=formula, data=data, topics=topics, ci=ci,
        cluster=cluster, weights=weights, link="identity", corpus=corpus,
        nsims=nsims, seed=seed, add_intercept=add_intercept,
    )

    # Capture training knots/factor encoding so perturbed designs match the fit.
    knot_ctx = _KnotCapturingContext()
    design_matrix(formula, data, _knot_ctx=knot_ctx)

    def _design(df):
        Xd, _ = design_matrix_predict(formula, df, knot_ctx)
        if add_intercept:
            Xd = np.hstack([np.ones((Xd.shape[0], 1)), Xd])
        return Xd

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        def _wmean(M):
            return (w[:, None] * M).sum(axis=0) / w.sum()
    else:
        def _wmean(M):
            return M.mean(axis=0)

    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    col = data[covariate]
    is_factor = (not is_numeric_dtype(col)) or is_bool_dtype(col)

    contrasts = {}  # term name -> averaged design-change vector
    if is_factor:
        levels = sorted(map(str, col.dropna().unique()))
        ref = levels[0]
        d_ref = _design(data.assign(**{covariate: ref}))
        for lv in levels[1:]:
            d_lv = _design(data.assign(**{covariate: lv}))
            contrasts[f"{covariate}{lv}"] = _wmean(d_lv - d_ref)
    else:
        x = np.asarray(col, dtype=np.float64)
        hh = h if h is not None else 0.01 * float(np.std(x))
        if hh <= 0:
            raise ValueError("covariate has zero variance; cannot form a derivative.")
        d_plus = _design(data.assign(**{covariate: x + hh}))
        d_minus = _design(data.assign(**{covariate: x - hh}))
        contrasts[covariate] = _wmean((d_plus - d_minus) / (2.0 * hh))

    z_crit = _normal_ppf(0.5 + ci / 2.0)
    rows = []
    for eff in effects:
        coef = np.asarray(eff.coef, dtype=np.float64)
        Sigma = np.asarray(eff.vcov, dtype=np.float64)
        tname = getattr(eff, "topic_name", None) or f"topic_{eff.topic}"
        for term, cv in contrasts.items():
            est = float(cv @ coef)
            var = float(cv @ Sigma @ cv)
            se = float(np.sqrt(max(var, 0.0)))
            rows.append(MarginalEffect(
                topic=eff.topic,
                topic_name=tname,
                term=term,
                ame=est,
                se=se,
                ci_low=est - z_crit * se,
                ci_high=est + z_crit * se,
            ))
    return AverageMarginalEffects(covariate=covariate, effects=rows)


# Short alias matching faSTM's `ame()`.
ame = average_marginal_effects


@dataclass
class PredictedPrevalence:
    """Predicted topic prevalence at a covariate grid, with simulation-based CIs.

    Produced by :func:`predicted_prevalence`. Each entry covers one topic across
    all grid points (for ``at``/``continuous``) or the contrast between two
    settings (for ``contrast``).

    Attributes
    ----------
    topic : int
        Zero-based topic index.
    topic_name : str
        Human-readable label (``topic_names`` from the model, or ``"topic_k"``).
    mode : str
        One of ``"at"``, ``"contrast"``, or ``"continuous"``.
    grid : list
        Reference covariate values: a list of dicts for ``at`` / ``continuous``
        (one per grid row), or ``[setting_a, setting_b]`` for ``contrast``.
    estimate : np.ndarray
        Mean predicted prevalence (or contrast), one entry per grid point.
    ci_low : np.ndarray
        Lower bound of the ``ci``-level simulation interval.
    ci_high : np.ndarray
        Upper bound.
    covariate : str or None
        For ``continuous``, the name of the swept covariate (convenient for
        plotting); ``None`` otherwise.

    Notes
    -----
    ``estimate``/``ci_low``/``ci_high`` are always arrays, one entry per grid
    point, even for a ``contrast`` (which has a single point). For that single
    -point case use the scalar helpers ``value`` and ``ci`` — ``f"{pp.value:+.3f}"``
    formats, whereas ``f"{pp.estimate:+.3f}"`` raises because ``estimate`` is an
    array. For every case, ``to_frame()`` gives a tidy per-point DataFrame.
    """

    topic: int
    topic_name: str
    mode: str
    grid: list
    estimate: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    covariate: str | None = None

    @property
    def value(self) -> float:
        """The single predicted value as a float. Valid when there is exactly one
        grid point (every ``contrast``, or a one-row ``at``); raises otherwise,
        directing you to ``estimate`` (the array) or ``to_frame()``."""
        est = np.atleast_1d(self.estimate)
        if est.size != 1:
            raise ValueError(
                f"value is only defined for a single grid point; this result has "
                f"{est.size}. Use estimate (the array, one per grid point) or "
                "to_frame()."
            )
        return float(est[0])

    @property
    def ci(self) -> tuple:
        """The single ``(ci_low, ci_high)`` as floats, under the same one-grid
        -point condition as :attr:`value`."""
        lo, hi = np.atleast_1d(self.ci_low), np.atleast_1d(self.ci_high)
        if lo.size != 1:
            raise ValueError(
                f"ci is only defined for a single grid point; this result has "
                f"{lo.size}. Use ci_low/ci_high (arrays) or to_frame()."
            )
        return float(lo[0]), float(hi[0])

    def to_frame(self):
        """Return a tidy pandas DataFrame with one row per grid point.

        Columns are ``topic``, ``topic_name``, any covariate column(s),
        ``estimate``, ``ci_low``, and ``ci_high``.
        """
        import pandas as pd

        rows = []
        for idx, (est, lo, hi) in enumerate(
            zip(self.estimate, self.ci_low, self.ci_high)
        ):
            row: dict = {
                "topic": self.topic,
                "topic_name": self.topic_name,
            }
            if self.mode == "contrast":
                row["contrast"] = str(self.grid[idx]) if idx < len(self.grid) else ""
            else:
                g = self.grid[idx] if idx < len(self.grid) else {}
                if isinstance(g, dict):
                    row.update(g)
                else:
                    row["value"] = g
            row["estimate"] = float(est)
            row["ci_low"] = float(lo)
            row["ci_high"] = float(hi)
            rows.append(row)
        return pd.DataFrame(rows)


def _build_reference_rows(
    at,
    contrast,
    continuous,
    data,
    formula,
    feature_names,
    X_train,
    knot_ctx,
    npoints,
    add_intercept,
):
    """Build the design rows ``X_new`` for the prediction grid.

    Returns ``(X_new (G, p), grid_labels, covariate_name_or_None)``.
    ``X_new`` already includes the intercept column when ``add_intercept``.
    """
    import pandas as pd
    from .formulas import design_matrix_predict

    if continuous is not None:
        # Sweep the named continuous covariate over its observed range.
        if data is None:
            raise ValueError("continuous= requires data= (the training DataFrame).")
        col = continuous
        x_obs = np.asarray(data[col], dtype=np.float64)
        grid_vals = np.linspace(x_obs.min(), x_obs.max(), npoints)
        # Hold all other numeric columns at their means, categoricals at their mode.
        ref = {}
        for c in data.columns:
            if c == col:
                continue
            col_vals = data[c]
            try:
                ref[c] = float(col_vals.mean())
            except (TypeError, AttributeError):
                ref[c] = col_vals.mode()[0] if len(col_vals) > 0 else col_vals.iloc[0]
        rows = []
        for v in grid_vals:
            row_dict = dict(ref)
            row_dict[col] = v
            rows.append(row_dict)
        grid_df = pd.DataFrame(rows)
        if formula is not None:
            X_new, _ = design_matrix_predict(formula, grid_df, knot_ctx)
        else:
            # Raw X path: only the swept column changed; rebuild with the same columns.
            if feature_names is None:
                raise ValueError(
                    "continuous= with raw X requires feature_names= to identify the column."
                )
            fn = list(feature_names)
            if col not in fn:
                raise ValueError(f"continuous={col!r} not in feature_names {fn}")
            ci_idx = fn.index(col)
            # Hold other columns at their column means.
            col_means = X_train.mean(axis=0)
            X_new = np.tile(col_means, (npoints, 1))
            X_new[:, ci_idx] = grid_vals
        if add_intercept:
            X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        grid_labels = [{col: float(v)} for v in grid_vals]
        return X_new, grid_labels, col

    if contrast is not None:
        # Two covariate settings; result is the difference (setting_b - setting_a).
        if isinstance(contrast, dict):
            if len(contrast) != 1:
                raise ValueError(
                    "contrast= as a dict must have exactly one key: "
                    "{covariate: [value_a, value_b]}."
                )
            col, vals = next(iter(contrast.items()))
            if len(vals) != 2:
                raise ValueError(
                    "contrast= dict value must be a list of two levels, e.g. "
                    '{"party": ["D", "R"]}.'
                )
            setting_a, setting_b = vals
        elif len(contrast) == 2:
            setting_a, setting_b = contrast
            col = None  # no named column in the sequence form
        else:
            raise ValueError(
                "contrast= must be either a one-key dict mapping a covariate to its "
                'two levels, {"party": ["D", "R"]} (reports level_b - level_a), or a '
                "2-element sequence of covariate settings, (setting_a, setting_b). "
                f"Got {contrast!r} (length {len(contrast)})."
            )

        def _single_row(setting):
            """Build one design row for a covariate setting (dict or scalar)."""
            if data is not None and formula is not None:
                if isinstance(setting, dict):
                    base = {c: (data[c].mean() if data[c].dtype.kind in "fc" else data[c].mode()[0])
                            for c in data.columns}
                    base.update(setting)
                elif col is not None:
                    # scalar contrast value paired with the named column from the
                    # dict form: {col: [val_a, val_b]}
                    base = {c: (data[c].mean() if data[c].dtype.kind in "fc" else data[c].mode()[0])
                            for c in data.columns}
                    base[col] = setting
                else:
                    raise ValueError(
                        "contrast= as a 2-element sequence requires each element "
                        "to be a dict of covariate settings, e.g. "
                        "contrast=({'party': 'D', 'year': 2012}, {'party': 'R', 'year': 2012}). "
                        "To contrast one variable use the dict form: "
                        "contrast={'party': ['D', 'R']}."
                    )
                row_df = pd.DataFrame([base])
                X_row, _ = design_matrix_predict(formula, row_df, knot_ctx)
            elif feature_names is not None:
                fn = list(feature_names)
                col_means = X_train.mean(axis=0)
                x_row = col_means.copy()
                if isinstance(setting, dict):
                    for k, v in setting.items():
                        if k in fn:
                            x_row[fn.index(k)] = v
                elif col is not None:
                    if col in fn:
                        x_row[fn.index(col)] = setting
                else:
                    # 2-element sequence with raw X: treat the sequence as
                    # (value_for_feature_0, ...) — only valid for single-feature models.
                    if len(fn) == 1:
                        x_row[0] = float(setting)
                    else:
                        raise ValueError(
                            "contrast= as a 2-element scalar sequence is only supported "
                            "for single-feature models when using raw X. For multi-feature "
                            "models pass a dict: contrast={'feature_name': [val_a, val_b]}."
                        )
                X_row = x_row[None, :]
            else:
                raise ValueError(
                    "contrast= requires either (formula=, data=) or "
                    "(X=, feature_names=) to build reference rows."
                )
            return X_row  # (1, p_raw)

        Xa = _single_row(setting_a)
        Xb = _single_row(setting_b)
        if add_intercept:
            Xa = np.hstack([np.ones((1, 1)), Xa])
            Xb = np.hstack([np.ones((1, 1)), Xb])
        # Stack both rows; the caller computes the difference.
        X_new = np.vstack([Xa, Xb])
        grid_labels = [str(setting_a), str(setting_b)]
        return X_new, grid_labels, None

    if at is not None:
        # Explicit reference grid.
        if isinstance(at, dict):
            # Could be {col: value} (single row) or {col: [v1, v2, ...]} (grid).
            vals = list(at.values())
            if any(isinstance(v, (list, np.ndarray)) for v in vals):
                # Convert to a list of dicts: one per combination, iterating
                # over the first list-valued covariate and broadcasting scalars.
                list_vals = {k: (v if isinstance(v, (list, np.ndarray)) else [v])
                             for k, v in at.items()}
                max_len = max(len(v) for v in list_vals.values())
                at_rows = []
                for i in range(max_len):
                    at_rows.append({k: (v[i % len(v)] if isinstance(v, (list, np.ndarray)) else v)
                                    for k, v in at.items()})
            else:
                at_rows = [at]
        elif hasattr(at, "iterrows"):
            # pandas DataFrame
            at_rows = [dict(row) for _, row in at.iterrows()]
        else:
            at_rows = list(at)

        X_parts = []
        for row_dict in at_rows:
            if data is not None and formula is not None:
                base = {c: (data[c].mean() if data[c].dtype.kind in "fc"
                            else data[c].mode()[0])
                        for c in data.columns}
                base.update(row_dict)
                row_df = pd.DataFrame([base])
                X_row, _ = design_matrix_predict(formula, row_df, knot_ctx)
            elif feature_names is not None:
                fn = list(feature_names)
                col_means = X_train.mean(axis=0)
                x_row = col_means.copy()
                for k, v in row_dict.items():
                    if k in fn:
                        x_row[fn.index(k)] = v
                X_row = x_row[None, :]
            else:
                raise ValueError(
                    "at= requires either (formula=, data=) or (X=, feature_names=)."
                )
            X_parts.append(X_row)
        X_new = np.vstack(X_parts)
        if add_intercept:
            X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        grid_labels = at_rows
        return X_new, grid_labels, None

    raise ValueError("One of at=, contrast=, or continuous= must be supplied.")


def predicted_prevalence(
    model,
    *,
    X=None,
    formula=None,
    data=None,
    feature_names=None,
    at=None,
    contrast=None,
    continuous=None,
    npoints=50,
    topics=None,
    link="identity",
    ci=0.95,
    nsims=25,
    n_sim=2000,
    corpus=None,
    seed=0,
    add_intercept=True,
):
    """Predicted topic prevalence at chosen covariate values, with simulation-based CIs.

    This is the model-agnostic counterpart of R ``stm``'s ``plot.estimateEffect``.
    It works on any model whose document-topic matrix supports
    :func:`~topica.effects.composition_theta` (STM, CTM, LDA, keyATM covariate,
    DMR, SeededLDA, ...) because it regresses the composition-theta draws on the
    design matrix — exactly as :func:`estimate_effect` does — and then pushes
    coefficient posterior draws through the link at new covariate values rather
    than reporting the coefficients themselves.

    Three modes mirror ``stm``'s ``method`` argument:

    - ``at=`` (**point grid**) — a dict ``{covariate: value}`` or a small DataFrame
      of reference rows; returns predicted theta per topic per row, with CI.
    - ``contrast=`` (**difference**) — two covariate settings, e.g.
      ``contrast={"party": ["D", "R"]}``; returns the difference in predicted
      theta between the two settings per topic, with CI.
    - ``continuous=`` (**smooth curve**) — a column name; sweeps the covariate
      over its observed range on a ``npoints``-point grid, holding all other
      columns at their means. Spline terms in ``formula`` are evaluated with the
      training knots, not re-fit to the new grid.

    Parameters
    ----------
    model : fitted topica model
        Any model whose theta supports the composition method (Gibbs or
        logistic-normal). Pass the model itself; theta draws are generated
        internally.
    X : array (num_docs, p), optional
        Raw design matrix. Provide either ``X`` (with optional ``feature_names``)
        or ``formula`` + ``data``.
    formula : str, optional
        R-style formula, e.g. ``"~ party + spline(year, df=3)"``.
    data : pandas.DataFrame, optional
        One row per document; required with ``formula=``. Also used to build
        reference rows for ``continuous=`` / ``contrast=``.
    feature_names : list[str], optional
        Column names for ``X``. Required for ``continuous=`` or ``contrast=``
        when using the raw ``X`` path.
    at : dict or DataFrame, optional
        Reference covariate settings for point predictions.
    contrast : dict or 2-element sequence, optional
        Two covariate settings; the result is their difference (second minus
        first). Two accepted forms: a one-key dict mapping a covariate to its two
        levels, ``{"party": ["D", "R"]}`` (reports ``R - D``); or a 2-element
        sequence of full settings, ``(setting_a, setting_b)``. A 3-tuple such as
        ``("party", "D", "R")`` is *not* accepted — use the dict form.
    continuous : str, optional
        Column name to sweep over its observed range.
    npoints : int
        Number of grid points for ``continuous=``. Default 50.
    topics : list[int], optional
        Restrict to these topics. Defaults to all.
    link : str
        ``"identity"`` (default), ``"logit"``, or ``"log"``. Applied to the
        linear predictor when computing predicted prevalence.
    ci : float
        Confidence level for the simulation-based interval. Default 0.95.
    nsims : int
        Composition theta draws for Rubin's-rules pooling. Default 25.
    n_sim : int
        Number of coefficient posterior draws for the simulation CI. Default 2000.
    corpus : Corpus or token lists, optional
        Required for Gibbs models that did not retain ``theta_draws``.
    seed : int
        RNG seed.
    add_intercept : bool
        Prepend an intercept column to the design matrix. Default True.

    Returns
    -------
    list[PredictedPrevalence]
        One object per topic (in ``topics`` order, or all topics). Each has
        ``.estimate``, ``.ci_low``, ``.ci_high`` arrays (one entry per grid
        point) and a ``.to_frame()`` method for a tidy DataFrame.
    """
    from .effects import composition_theta
    from .formulas import _KnotCapturingContext

    # --- build training design matrix ----------------------------------------
    knot_ctx = _KnotCapturingContext()
    if formula is not None:
        if data is None:
            raise ValueError("formula= requires data= (a pandas DataFrame).")
        from .formulas import design_matrix
        X_train, feature_names = design_matrix(formula, data, _knot_ctx=knot_ctx)
    elif X is not None:
        X_train = np.asarray(X, dtype=np.float64)
        if X_train.ndim == 1:
            X_train = X_train[:, None]
    else:
        raise ValueError("provide X (a design matrix), or formula= with data=.")

    # --- draw theta ----------------------------------------------------------
    theta = composition_theta(model, corpus, nsims=nsims, seed=seed)  # (M, D, K)
    m, n, num_topics = theta.shape
    if X_train.shape[0] != n:
        raise ValueError(
            f"X has {X_train.shape[0]} rows but the model's doc_topic has {n} docs"
        )

    names = list(feature_names) if feature_names is not None else [
        f"feature_{i}" for i in range(X_train.shape[1])
    ]
    if len(names) != X_train.shape[1]:
        raise ValueError("feature_names length must match X columns")

    if link not in ("identity", "logit", "log"):
        raise ValueError("link must be 'identity', 'logit', or 'log'")

    # Add intercept to training matrix.
    X_full = np.hstack([np.ones((n, 1)), X_train]) if add_intercept else X_train
    names_full = (["intercept"] + names) if add_intercept else names

    p = X_full.shape[1]
    XtX_inv = np.linalg.pinv(X_full.T @ X_full)
    hat = XtX_inv @ X_full.T
    dof = max(n - p, 1)

    topic_list = list(range(num_topics)) if topics is None else list(topics)
    for t in topic_list:
        if t < 0 or t >= num_topics:
            raise ValueError(f"topic {t} out of range (num_topics={num_topics})")

    # --- get per-topic coefficient posterior ---------------------------------
    pooled = _pooled_coefficients(
        theta, X_full, link=link, groups=None, hat=hat, XtX_inv=XtX_inv, dof=dof,
        topic_list=topic_list,
    )

    # --- build reference design rows X_new ----------------------------------
    X_new, grid_labels, cov_name = _build_reference_rows(
        at=at,
        contrast=contrast,
        continuous=continuous,
        data=data,
        formula=formula,
        feature_names=names,
        X_train=X_train,
        knot_ctx=knot_ctx,
        npoints=npoints,
        add_intercept=add_intercept,
    )
    # X_new already has intercept prepended by _build_reference_rows.
    G = X_new.shape[0]

    # --- simulation-based CI ------------------------------------------------
    rng = np.random.default_rng(seed)
    mode = "contrast" if contrast is not None else ("continuous" if continuous is not None else "at")

    # Topic names
    topic_names_all = list(getattr(model, "topic_names", [])) or [
        f"topic_{t}" for t in range(num_topics)
    ]

    alpha = 1.0 - ci
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    out: list[PredictedPrevalence] = []
    for (beta, Sigma, _r2), t in zip(pooled, topic_list):
        # Symmetrise and regularise Sigma for Cholesky.
        Sigma_sym = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(p)
        try:
            L = np.linalg.cholesky(Sigma_sym)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(Sigma_sym)
            L = v @ np.diag(np.sqrt(np.clip(w, 0.0, None)))

        # Draw n_sim coefficient vectors from the posterior N(beta, Sigma).
        Z = rng.standard_normal((n_sim, p))
        beta_draws = beta[None, :] + Z @ L.T  # (n_sim, p)

        # Predicted prevalence at each grid point.
        eta = beta_draws @ X_new.T  # (n_sim, G)
        pred = _link_inv(eta, link)  # (n_sim, G)

        if mode == "contrast":
            # Difference: setting_b (row 1) minus setting_a (row 0).
            diff = pred[:, 1] - pred[:, 0]  # (n_sim,)
            estimates = np.array([float(diff.mean())])
            ci_lo = np.array([float(np.percentile(diff, q_lo * 100))])
            ci_hi = np.array([float(np.percentile(diff, q_hi * 100))])
        else:
            estimates = pred.mean(axis=0)
            ci_lo = np.percentile(pred, q_lo * 100, axis=0)
            ci_hi = np.percentile(pred, q_hi * 100, axis=0)

        tname = topic_names_all[t] if t < len(topic_names_all) else f"topic_{t}"
        grid_out = [grid_labels[0], grid_labels[1]] if mode == "contrast" else grid_labels
        out.append(PredictedPrevalence(
            topic=t,
            topic_name=tname,
            mode=mode,
            grid=grid_out,
            estimate=estimates,
            ci_low=ci_lo,
            ci_high=ci_hi,
            covariate=cov_name,
        ))
    return out


def posterior_theta_samples(model, nsims=25, seed=0, *, uncertainty="local"):
    """Draw `nsims` samples of the document-topic matrix θ from a fitted
    :class:`STM`/:class:`CTM`'s variational posterior.

    Each document's logistic-normal posterior is centered at ``λ_d``
    (``model.eta_mean``); a draw of η is mapped through the softmax (with the
    reference category fixed at 0) to a θ row. Feed the result to
    :func:`estimate_effect` for method-of-composition uncertainty.

    ``uncertainty`` chooses the draw covariance, matching R ``stm``'s
    ``thetaPosterior`` ``type=``:

    - ``"local"`` (default here) — each document draws from its own variational
      covariance ``ν_d`` (``model.eta_cov``). This is R's ``type="Local"``.
    - ``"global"`` — every document draws from one *shared* covariance, R's
      ``type="Global"`` (``Σ − crossprod(λ − μ)/N``). For STM/CTM's M-step that
      shared covariance is identically the mean per-document variational
      covariance ``mean_d(ν_d)`` (the between-document spread of the point λ is
      exactly what the ``Σ`` update adds to ``mean(ν)``), so this propagates the
      global topic-model uncertainty rather than each document's local Hessian.
      It is R ``estimateEffect``'s default; :func:`estimate_effect` uses it.
    - ``"none"`` — no topic-model uncertainty: every draw is the point θ
      (``model.doc_topic``). Method-of-composition then reduces to OLS on the
      point estimate and *understates* uncertainty; provided for parity with R's
      ``type="None"`` and for a fast point-estimate pass.

    Returns an array of shape ``(nsims, num_docs, num_topics)``.
    """
    if uncertainty not in ("local", "global", "none"):
        raise ValueError(
            f"uncertainty must be 'local', 'global', or 'none' (got {uncertainty!r})"
        )
    # This is the logistic-normal (STM/CTM) sampler. Refuse other families cleanly
    # rather than failing with an AttributeError on the missing ``eta_mean`` getter
    # (issue #651): point at the right path for each family.
    from .effects import model_family

    fam = model_family(model)
    if fam != "logistic_normal":
        if fam == "dirichlet":
            raise ValueError(
                f"{type(model).__name__} has a Dirichlet (not logistic-normal) "
                "posterior; use topica.composition_theta(model, corpus) to draw "
                "theta for method-of-composition, not posterior_theta_samples."
            )
        raise ValueError(
            f"{type(model).__name__} has no posterior over topic proportions, so "
            "theta cannot be sampled. Use standard_errors(model, corpus, "
            "of='effect', method='bootstrap') for uncertainty on this model."
        )
    lam = np.asarray(model.eta_mean, dtype=np.float64)  # (D, K-1)
    d, km1 = lam.shape

    # "none": no topic-model uncertainty — every draw is the point θ.
    if uncertainty == "none":
        full = np.concatenate([lam, np.zeros((d, 1))], axis=1)  # ref cat = 0
        full -= full.max(axis=1, keepdims=True)
        e = np.exp(full)
        theta = e / e.sum(axis=1, keepdims=True)                # (D, K)
        return np.broadcast_to(theta, (nsims, d, theta.shape[1])).copy()

    try:
        cov = np.asarray(model.eta_cov, dtype=np.float64)   # (D, K-1, K-1)
    except RuntimeError:
        cov = np.asarray(model._recompute_eta_cov(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    eye = np.eye(km1)

    if uncertainty == "global":
        # One shared covariance for all documents: the mean per-doc variational
        # covariance, which equals R's Global Σ − crossprod(λ − μ)/N (the STM/CTM
        # M-step sets Σ = crossprod(λ − μ)/N + mean(ν) at sigma.prior=0). Averaging
        # PSD covariances keeps it PSD, so the Cholesky is robust.
        shared = 0.5 * (cov.mean(axis=0) + cov.mean(axis=0).T) + 1e-10 * eye
        try:
            l_shared = np.linalg.cholesky(shared)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(shared)
            l_shared = v @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
        z = rng.standard_normal((d, nsims, km1))
        eta = lam[:, None, :] + z @ l_shared.T               # (D, nsims, K-1)
    else:
        # Local: per-document covariance ν_d. Cholesky is all-or-nothing on a
        # batch, so only fall back to per-doc eigh for the docs that aren't PD —
        # the common (all-PD) case stays a single batched LAPACK call.
        csym = 0.5 * (cov + cov.transpose(0, 2, 1)) + 1e-10 * eye
        try:
            chol = np.linalg.cholesky(csym)                  # (D, K-1, K-1)
        except np.linalg.LinAlgError:
            chol = np.empty_like(csym)
            for di in range(d):
                try:
                    chol[di] = np.linalg.cholesky(csym[di])
                except np.linalg.LinAlgError:
                    w, v = np.linalg.eigh(csym[di])
                    chol[di] = v @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
        # Draw in document order (matches the old per-doc loop's RNG stream), then
        # η = λ + Z·Lᵀ for all docs/sims via one batched matmul.
        z = rng.standard_normal((d, nsims, km1))
        eta = lam[:, None, :] + z @ chol.transpose(0, 2, 1)  # (D, nsims, K-1)

    full = np.concatenate([eta, np.zeros((d, nsims, 1))], axis=2)  # ref cat = 0
    full -= full.max(axis=2, keepdims=True)
    e = np.exp(full)
    theta = e / e.sum(axis=2, keepdims=True)             # (D, nsims, K)
    return theta.transpose(1, 0, 2).copy()               # (nsims, D, K)


@dataclass
class TopicCorrelationCI:
    """A topic-correlation matrix with a credible interval per cell.

    ``estimate``/``se``/``ci_low``/``ci_high`` are each ``(num_topics,
    num_topics)`` arrays: the point estimate (the model's ``topic_correlation``),
    the posterior standard deviation, and the lower/upper credible bounds.
    """

    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _theta_correlation(theta):
    """Across-document correlation of a doc-topic matrix ``theta`` (D, K) -> (K, K),
    matching the core ``topic_correlation`` (biased covariance, unit diagonal, zero
    where a topic has no variance)."""
    cov = np.cov(np.asarray(theta, dtype=np.float64), rowvar=False, bias=True)
    cov = np.atleast_2d(cov)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    den = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        cor = np.where(den > 0.0, cov / den, 0.0)
    np.fill_diagonal(cor, 1.0)
    return cor


def topic_correlation_ci(model, *, nsims=200, ci=0.9, seed=0):
    """Credible interval on an STM/CTM topic-correlation matrix.

    ``model.topic_correlation`` is the across-document correlation of the
    *posterior-mean* doc-topic matrix. This propagates the model's logistic-normal
    posterior into that statistic: it draws ``nsims`` doc-topic matrices from the
    per-document posterior ``η_d ~ N(λ_d, ν_d)`` (via
    :func:`posterior_theta_samples`), recomputes the correlation on each draw, and
    reports the per-cell posterior mean, SD, and percentile interval. A cell whose
    interval straddles zero is not a reliably signed topic relationship.

    The returned ``estimate`` is the posterior *mean* of the across-draw
    correlations, so the interval is centered on it. This is deliberately distinct
    from ``model.topic_correlation``: the latter correlates the posterior means and
    so ignores within-document posterior variance, which makes it more extreme than
    the uncertainty-aware draw-based estimate here (adding per-document posterior
    noise attenuates an across-document correlation). Use ``estimate`` with its
    interval when you need well-calibrated uncertainty; ``model.topic_correlation`` remains
    the conventional point summary.

    Reuses the same posterior draws as method-of-composition effect estimation, so
    it adds no new model state. Only logistic-normal models (STM/CTM/STS, which
    carry ``eta_mean``/``eta_cov``) are supported.

    Parameters
    ----------
    model : STM or CTM
        A fitted logistic-normal topic model.
    nsims : int
        Number of posterior draws.
    ci : float
        Central interval mass (default 0.9 for a 90% interval).
    seed : int
        Seed for the posterior draws.

    Returns
    -------
    TopicCorrelationCI
        ``(estimate, se, ci_low, ci_high)``, each ``(num_topics, num_topics)``.
    """
    cls = type(model)
    if not (hasattr(cls, "eta_mean") and hasattr(cls, "eta_cov")):
        raise TypeError(
            "topic_correlation_ci requires a logistic-normal model (STM/CTM/STS) "
            "with eta_mean/eta_cov; this model exposes neither"
        )
    draws = posterior_theta_samples(model, nsims=nsims, seed=seed)  # (nsims, D, K)
    cors = np.stack([_theta_correlation(draws[s]) for s in range(draws.shape[0])])
    estimate = cors.mean(axis=0)
    np.fill_diagonal(estimate, 1.0)
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    ci_low = np.quantile(cors, lo_q, axis=0)
    ci_high = np.quantile(cors, hi_q, axis=0)
    se = cors.std(axis=0, ddof=1) if cors.shape[0] > 1 else np.zeros_like(estimate)
    return TopicCorrelationCI(estimate, se, ci_low, ci_high)


def spline(x, df=4, knots=None):
    """Restricted (natural) cubic-spline basis for a covariate — the building
    block for nonlinear prevalence terms like R ``stm``'s ``~ s(day)``.

    Uses Harrell's restricted-cubic-spline parameterization: `df+1` knots (at
    evenly spaced quantiles of `x` unless `knots` is given) yield `df` basis
    columns whose first is the linear term. ``np.column_stack`` the result into
    your design matrix and extend ``feature_names`` with the returned names.

    Returns ``(basis (n, df), names)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if df < 2:
        raise ValueError("spline df must be >= 2")
    if knots is None:
        knots = np.quantile(x, np.linspace(0.0, 1.0, df + 1))
    t = np.asarray(knots, dtype=np.float64)
    k = len(t)
    if k < 3:
        raise ValueError("need at least 3 knots (df >= 2)")
    denom = (t[-1] - t[0]) ** 2

    def cube(u):
        return np.clip(u, 0.0, None) ** 3

    cols = [x]
    for j in range(k - 2):
        term = (
            cube(x - t[j])
            - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        ) / denom
        cols.append(term)
    basis = np.column_stack(cols)
    names = ["spline_lin"] + [f"spline_{j + 1}" for j in range(basis.shape[1] - 1)]
    return basis, names


def interaction(a, b, name="interaction"):
    """Interaction columns between two covariate blocks (all pairwise products of
    their columns) — for terms like R ``stm``'s ``~ treatment * party``.

    `a`, `b` are 1-D or 2-D arrays with the same number of rows. Returns
    ``(products (n, ncols), names)``; ``np.column_stack`` into your design matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same number of rows")
    cols = []
    names = []
    multi = a.shape[1] > 1 or b.shape[1] > 1
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            cols.append(a[:, i] * b[:, j])
            names.append(f"{name}_{i}x{j}" if multi else name)
    return np.column_stack(cols), names


def _normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = (-2 * np.log(q)) ** 0.5
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / (
            (((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > phigh:
        r = (-2 * np.log(1 - q)) ** 0.5
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / (
            (((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    s = r * r
    return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / (
        ((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


# ---------------------------------------------------------------------------
# align_corpus: vocabulary alignment for out-of-sample documents
# ---------------------------------------------------------------------------

def align_corpus(new_docs, model):
    """Restrict token lists to the fitted model's vocabulary before transform.

    Each document in `new_docs` is filtered to keep only tokens that appear in
    ``model.vocabulary``. Tokens outside that vocabulary are silently dropped.
    Documents that become empty after filtering are represented as empty lists.

    Parameters
    ----------
    new_docs : list[list[str]]
        Token lists for the new documents (one list per document).
    model : fitted STM or CTM
        A fitted model with a ``vocabulary`` attribute (list of strings).

    Returns
    -------
    list[list[str]]
        Aligned token lists ready to pass to ``model.transform`` or
        ``topica.stm.transform``. Each output list is a subset of the
        corresponding input list, with out-of-vocabulary tokens removed.
    """
    vocab_set = set(model.vocabulary)
    return [[tok for tok in doc if tok in vocab_set] for doc in new_docs]


# ---------------------------------------------------------------------------
# transform: covariate-aware out-of-sample inference for STM
# ---------------------------------------------------------------------------

def transform(model, docs, *, prevalence=None, data=None, formula=None, X=None):
    """Infer topic proportions for new documents, optionally using prevalence covariates.

    When prevalence information is supplied the per-document prior mean is set
    to ``mu_d = X_d @ gamma`` (where ``gamma = model.prevalence_effects``),
    which mirrors R ``stm``'s ``fitNewDocuments`` behavior. Without covariates
    the covariate-free baseline prior learned at fit time is used, giving the
    same result as ``model.transform(docs)`` directly.

    The topic-word matrix used is always the marginal ``model.topic_word``; a
    content model's per-group beta is not applied here. Documents should first
    be aligned to the fitted vocabulary with :func:`align_corpus` if the new
    corpus may contain out-of-vocabulary tokens.

    Parameters
    ----------
    model : fitted STM
        A fitted ``topica.STM`` with ``prevalence_effects`` available when
        covariates are supplied.
    docs : list[list[str]] or Corpus
        Token lists (or a Corpus) for the new documents.
    prevalence : array-like (num_docs, F), optional
        Raw covariate matrix for the new documents, without the intercept
        column. An intercept is prepended to match how ``gamma`` was learned.
        Supply either ``prevalence`` or ``X``; they are equivalent.
    data : pandas.DataFrame, optional
        Document-level DataFrame for the new documents. Required when
        ``formula`` is given.
    formula : str, optional
        R-style formula string (e.g. ``"~ party + author"``). When supplied with
        ``data``, the design matrix is built from the formula using the same
        column encoding as at fit time (categorical coding, intercept stripping);
        an intercept is then prepended so the column order matches ``gamma``.
        Formulas with a ``spline()`` term are rejected here, because their knots
        would be recomputed on the new documents rather than reused from fit;
        build the design with ``design_matrix_predict`` and the fit-time knot
        context (as :func:`predicted_prevalence` does) and pass it as ``X=``.
    X : array-like (num_docs, p), optional
        Pre-built design matrix without the intercept column. Alternative to
        ``prevalence``; they are equivalent.

    Returns
    -------
    numpy.ndarray
        Topic proportions, shape ``(num_docs, num_topics)``.
    """
    # Accept prevalence= or X= as aliases for the raw matrix path.
    raw_x = prevalence if prevalence is not None else X

    if formula is not None or raw_x is not None:
        # Retrieve gamma from the model (raises RuntimeError if not fitted with
        # prevalence covariates).
        gamma = np.asarray(model.prevalence_effects, dtype=np.float64)  # (F, K-1)

        if formula is not None:
            if data is None:
                raise ValueError("formula= requires data= (a pandas DataFrame).")
            if "spline(" in formula:
                # A spline term's knots are placed from the training data at fit
                # time. Rebuilding the design from a bare formula here would
                # recompute knots on the new data, giving a silently miscalibrated
                # prior. Until the model retains its fit-time knot context, route
                # spline prevalence designs through the pre-built X path: build
                # X_new with design_matrix_predict and the training knot context
                # (as predicted_prevalence does), then pass it as X=.
                raise ValueError(
                    "formula= with a spline() term is not supported in transform "
                    "(its knots would be recomputed on the new documents rather "
                    "than reused from fit). Build the design matrix with "
                    "design_matrix_predict using the fit-time knot context and "
                    "pass it as X=."
                )
            from .formulas import design_matrix
            X_raw, _ = design_matrix(formula, data)
            X_raw = np.asarray(X_raw, dtype=np.float64)
        else:
            X_raw = np.asarray(raw_x, dtype=np.float64)
            if X_raw.ndim == 1:
                X_raw = X_raw[:, None]

        n = X_raw.shape[0]
        # Prepend intercept column to match how gamma was learned.
        X_full = np.hstack([np.ones((n, 1)), X_raw])  # (n, F)

        if X_full.shape[1] != gamma.shape[0]:
            raise ValueError(
                f"Design matrix (with intercept) has {X_full.shape[1]} columns "
                f"but gamma has {gamma.shape[0]} rows. Check that the number of "
                f"covariate columns matches the fitted model."
            )

        eta_prior_mean = X_full @ gamma  # (n, K-1)
        return model.transform(docs, eta_prior_mean=eta_prior_mean)

    # No covariates: fall back to the baseline prior.
    return model.transform(docs)


# ---------------------------------------------------------------------------
# Back-compatibility: the general post-hoc diagnostics were moved to
# ``topica.evaluate.diagnostics`` (they apply to any model, not just STM) and are
# also exported at the package top level. They are re-exported here so existing
# ``topica.stm.<name>`` calls keep working.
# ---------------------------------------------------------------------------
from .validation import (  # noqa: E402,F401
    frex,
    label_topics,
    topic_correlation,
    TopicCorrelation,
    find_thoughts,
    search_k,
    relevance,
    prepare_pyldavis,
    PyLDAvisInputs,
    check_residuals,
    ResidualCheck,
    align_topics,
    topic_stability,
)


__all__ = [
    "estimate_effect",
    "TopicEffect",
    "predicted_prevalence",
    "PredictedPrevalence",
    "posterior_theta_samples",
    "spline",
    "interaction",
    "align_corpus",
    "transform",
    "frex",
    "label_topics",
    "topic_correlation",
    "TopicCorrelation",
    "find_thoughts",
    "search_k",
    "relevance",
    "prepare_pyldavis",
    "PyLDAvisInputs",
    "check_residuals",
    "ResidualCheck",
    "align_topics",
    "topic_stability",
]
