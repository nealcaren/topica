"""R-style formula interface for building prevalence/effect design matrices.

Wraps `formulaic` (an optional dependency) so social scientists can write
``"~ treatment * party + spline(year, df=3)"`` instead of hand-stitching
`one_hot` and `np.hstack`. The library's own restricted-cubic-spline (the same
one behind :func:`topica.design.spline`) is exposed inside formulas as ``spline(...)``,
so the basis matches what the STM toolkit uses elsewhere.
"""

from __future__ import annotations

import numpy as np


class _KnotCapturingContext:
    """Wraps a formula evaluation to capture the knots used by each ``spline(...)``
    call during training, then provides a frozen context that re-applies those
    exact knots when evaluating the formula on a new grid.

    Also stores the ``formulaic.ModelSpec`` produced by the training
    ``design_matrix`` call.  :func:`design_matrix_predict` re-uses that spec
    (via ``spec.get_model_matrix``) so that categorical factor levels, dummy
    coding, and other encoding decisions stay consistent with training even
    when a single-row prediction frame contains only one level of a factor.

    Usage::

        kc = _KnotCapturingContext()
        X_train, names = design_matrix(formula, data, _knot_ctx=kc)
        X_new = kc.predict(formula, new_data)
    """

    def __init__(self):
        self._knots_by_order: dict[int, np.ndarray] = {}
        self._call_order: int = 0
        self.model_spec = None  # set by design_matrix after the training call

    def training_context(self) -> dict:
        """Return a formulaic ``context`` dict that records knots during training."""
        from .stm import spline as _spline

        ctx = self

        def spline(x, df=4, knots=None):
            basis, _ = _spline(x, df=df, knots=knots)
            # Record which knots were actually used (compute from x when not supplied).
            if knots is None:
                actual = np.quantile(np.asarray(x, dtype=np.float64),
                                     np.linspace(0.0, 1.0, df + 1))
            else:
                actual = np.asarray(knots, dtype=np.float64)
            ctx._knots_by_order[ctx._call_order] = actual
            ctx._call_order += 1
            return basis

        return {"spline": spline}

    def prediction_context(self) -> dict:
        """Return a formulaic ``context`` dict that re-uses training knots in order."""
        from .stm import spline as _spline

        ctx = self
        call_order = [0]

        def spline(x, df=4, knots=None):
            key = call_order[0]
            call_order[0] += 1
            frozen_knots = ctx._knots_by_order.get(key, knots)
            basis, _ = _spline(x, df=df, knots=frozen_knots)
            return basis

        return {"spline": spline}


def _formula_context():
    """Symbols available inside a formula beyond the data columns."""
    from .stm import spline as _spline

    def spline(x, df=4, knots=None):
        # formulaic names the columns spline(col, df=k)[0..]; we return just the
        # basis (drop the names tuple from topica.design.spline).
        return _spline(x, df=df, knots=knots)[0]

    return {"spline": spline}


def design_matrix(formula, data, _knot_ctx=None):
    """Build a design matrix from an R-style `formula` and a `data` frame
    (pandas or Polars).

    Returns ``(X, feature_names)`` where ``X`` is a ``(n_rows, p)`` float array
    and ``feature_names`` are the column labels. The intercept that `formulaic`
    adds is stripped, because :func:`topica.effects.estimate_effect` and the STM
    prevalence model add their own. Categorical columns become treatment-coded
    dummies; ``a * b`` / ``a:b`` expand interactions; ``spline(x, df=k)`` uses
    topica's restricted cubic spline. A Polars frame is converted to pandas for
    `formulaic`.

    Requires the optional ``formulaic`` package: install it with
    ``pip install "topica[formula]"`` (or ``pip install formulaic``). Without it,
    this raises ``ImportError`` at call time. If you would rather build the design
    matrix without that dependency, use :func:`topica.design.one_hot`,
    :func:`topica.design.spline`, and :func:`topica.design.interaction` directly.

    Parameters
    ----------
    formula : str
        R-style formula, e.g. ``"~ party + spline(year, df=3)"``.
    data : pandas.DataFrame or polars.DataFrame
        One row per document; columns referenced in ``formula`` must be present.
    _knot_ctx : _KnotCapturingContext, optional
        When supplied, the ``spline`` evaluations use the context's
        training mode so the knots are recorded.  Pass the same object
        to :func:`design_matrix_predict` to replay those knots on new data.
    """
    # The signature is formula-first: design_matrix("~ party", df). Passing the
    # data frame first gives formulaic an opaque failure, so catch it up front.
    if not isinstance(formula, str):
        raise TypeError(
            "design_matrix(formula, data): the first argument must be an R-style "
            f"formula string (e.g. '~ party + spline(year, df=3)'), got "
            f"{type(formula).__name__}. Did you pass the data frame first? The "
            "order is formula, then data."
        )
    try:
        from formulaic import model_matrix
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "The formula interface needs the optional `formulaic` package "
            "(pip install formulaic, or pip install \"topica[formula]\")."
        ) from e
    import pandas as pd

    from .frames import _is_polars

    if _is_polars(data):
        # Build the pandas frame from columns directly; data.to_pandas() would
        # pull in pyarrow, which we do not want to require.
        data = pd.DataFrame(data.to_dict(as_series=False))

    ctx = _knot_ctx.training_context() if _knot_ctx is not None else _formula_context()
    mm = model_matrix(formula, data, context=ctx)
    if _knot_ctx is not None:
        _knot_ctx.model_spec = mm.model_spec
    frame = pd.DataFrame(mm)
    if "Intercept" in frame.columns:
        frame = frame.drop(columns=["Intercept"])
    X = frame.to_numpy(dtype=float)
    names = [str(c) for c in frame.columns]
    return X, names


def design_matrix_predict(formula, data, knot_ctx):
    """Evaluate ``formula`` on ``data`` using the training knots captured in
    ``knot_ctx``.

    This is the companion to :func:`design_matrix` with ``_knot_ctx=``: call
    :func:`design_matrix` on the training frame, then call this function on the
    prediction grid to get a consistent design matrix where spline terms use the
    same knots as training.

    When ``knot_ctx.model_spec`` is available (set automatically by
    :func:`design_matrix`), prediction is done via
    ``spec.get_model_matrix(data, context=...)`` so that the encoding of
    categorical columns (treatment contrasts, factor levels) is fixed at the
    training schema.  This prevents single-row prediction frames from dropping
    dummy columns for factor levels not present in that row.

    Returns ``(X, feature_names)`` — same layout as :func:`design_matrix`.
    """
    try:
        from formulaic import model_matrix
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The formula interface needs the optional `formulaic` package "
            "(pip install formulaic, or pip install \"topica[formula]\")."
        ) from e
    import pandas as pd

    from .frames import _is_polars

    if _is_polars(data):
        data = pd.DataFrame(data.to_dict(as_series=False))

    ctx = knot_ctx.prediction_context()
    spec = getattr(knot_ctx, "model_spec", None)
    if spec is not None:
        # Re-use the training ModelSpec so categorical factor levels and dummy
        # coding stay consistent even when the prediction frame has only one
        # level of a factor.
        mm = spec.get_model_matrix(data, context=ctx)
    else:
        mm = model_matrix(formula, data, context=ctx)
    frame = pd.DataFrame(mm)
    if "Intercept" in frame.columns:
        frame = frame.drop(columns=["Intercept"])
    X = frame.to_numpy(dtype=float)
    names = [str(c) for c in frame.columns]
    return X, names
