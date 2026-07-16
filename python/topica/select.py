"""Choosing the topic count K and picking among fitted models.

Fit across a range of K and score each (:func:`search_k`), or rank a set of
already-fitted candidates and pick one (:func:`select_model`).
"""

from __future__ import annotations

from .validation import (
    search_k,
    SearchKResult,
    select_model,
    SelectModelResult,
)

__all__ = ["search_k", "SearchKResult", "select_model", "SelectModelResult"]
