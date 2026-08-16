"""Covariate design construction (workflow namespace, issue #757).

Build the design a covariate model reads: R-style formulas, one-hot encodings,
splines, and interactions. Re-exports the helpers that already live in
:mod:`topica.formulas` and :mod:`topica.stm`; the names are also available at the
package root (``topica.design_matrix``, ``topica.spline``, ``topica.one_hot`` …).
"""

from __future__ import annotations

from .formulas import design_matrix, design_matrix_predict
from .stm import spline, interaction, align_corpus

__all__ = [
    "one_hot",
    "design_matrix",
    "design_matrix_predict",
    "spline",
    "interaction",
    "align_corpus",
]


def one_hot(values, *, drop_first=True, reference=None, prefix=""):
    """One-hot encode a categorical covariate for use as DMR features.

    Given a sequence of category labels (one per document), returns
    ``(matrix, names)`` where ``matrix`` is a ``(num_docs, num_categories)``
    float array of 0/1 indicators and ``names`` are the corresponding column
    names. One category is omitted as the reference (baseline) level, which
    avoids collinearity with the DMR/STM intercept; every coefficient is then a
    contrast *against that reference*, so which level is the reference sets what
    the effects mean.

    ``reference`` names the level to drop. When ``None`` (and ``drop_first=True``,
    the default), the alphabetically-first category is dropped; with three or more
    levels a warning names it, since which of several baselines is the reference
    shapes the whole story and a silent choice is easy to misread — pass
    ``reference=`` to choose it explicitly (and silence the warning).
    ``drop_first=False`` keeps every level (full dummy set, e.g. for a model
    without an intercept). Pass the result straight to ``DMR.fit(docs, matrix,
    feature_names=names)``; combine multiple covariates with ``numpy.hstack``.
    """
    import numpy as np
    import warnings

    values = list(values)
    categories = sorted(set(values))
    if reference is not None:
        if reference not in categories:
            raise ValueError(
                f"reference={reference!r} is not one of the categories "
                f"{categories}"
            )
        categories = [c for c in categories if c != reference]
    elif drop_first and categories:
        dropped = categories[0]
        categories = categories[1:]
        # Binary covariate: the baseline is unambiguous (only one other level), so
        # warning would be noise. With 3+ levels the reference genuinely shapes the
        # contrasts and a silent alphabetical choice is a Tier-1 footgun.
        if len(categories) >= 2:
            warnings.warn(
                f"one_hot: dropped the alphabetically-first level {dropped!r} as "
                "the reference, so every coefficient is a contrast against it. "
                "Pass reference= to choose the baseline explicitly (and silence "
                "this warning), or drop_first=False to keep all levels.",
                UserWarning,
                stacklevel=2,
            )
    index = {c: j for j, c in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for i, v in enumerate(values):
        j = index.get(v)
        if j is not None:
            matrix[i, j] = 1.0
    names = [f"{prefix}{c}" for c in categories]
    return matrix, names
