"""Covariate design matrices for prevalence and content models.

Build the ``(num_docs, F)`` feature blocks that DMR, STM, STS, keyATM, and the
other covariate models take: one-hot encode a categorical, expand a spline basis,
form interactions, or parse an R-style formula. ``column_stack`` the pieces into
one design matrix and extend ``feature_names`` to match.
"""

from __future__ import annotations

import numpy as np

from .formulas import design_matrix
from .stm import spline, interaction

__all__ = ["one_hot", "design_matrix", "spline", "interaction"]


def one_hot(values, *, drop_first=True, prefix=""):
    """One-hot encode a categorical covariate for use as DMR features.

    Given a sequence of category labels (one per document), returns
    ``(matrix, names)`` where ``matrix`` is a ``(num_docs, num_categories)``
    float array of 0/1 indicators and ``names`` are the corresponding column
    names. With ``drop_first=True`` (default) the first category (sorted) is
    omitted as the reference level, which avoids collinearity with the DMR
    intercept. Pass the result straight to ``DMR.fit(docs, matrix,
    feature_names=names)``; combine multiple covariates with ``numpy.hstack``.
    """
    values = list(values)
    categories = sorted(set(values))
    if drop_first and categories:
        categories = categories[1:]
    index = {c: j for j, c in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for i, v in enumerate(values):
        j = index.get(v)
        if j is not None:
            matrix[i, j] = 1.0
    names = [f"{prefix}{c}" for c in categories]
    return matrix, names
