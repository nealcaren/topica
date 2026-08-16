"""Small result containers shared across the diagnostic helper families.

Several diagnostics (:func:`~topica.select.quality_frontier`,
:func:`~topica.evaluate.bootstrap_stability`, :func:`~topica.visualize_keywords`,
:func:`~topica.time_prevalence_ci`) historically returned a plain ``dict``. They
still *are* dicts — every ``result[key]`` access, ``.get``, ``**`` unpacking, and
JSON round-trip is unchanged, and ``FrameDict({...}) == {...}`` — but they now
carry the same ``.to_frame()`` a pandas user already reaches for on the other
result types (:class:`~topica.select.SearchKResult`, the effect/robustness results).
That converges the "return-type zoo" (#742) onto one idiom without breaking the
dict contract.

Each subclass defines its own natural *tidy* shape in :meth:`FrameDict.to_frame`;
the base frames every value as a column, which is correct when the values are
equal-length arrays.
"""

from __future__ import annotations


class FrameDict(dict):
    """A ``dict`` result that also renders as a tidy ``pandas.DataFrame``.

    Behaves exactly like the ``dict`` it subclasses; adds :meth:`to_frame`.
    """

    def to_frame(self):
        """Render as a ``pandas.DataFrame`` (raises if pandas is absent)."""
        import pandas as pd

        return pd.DataFrame(dict(self))


class QualityFrontier(FrameDict):
    """:func:`~topica.select.quality_frontier` result: one row per topic with
    ``topic``, ``coherence``, ``exclusivity``, ``prevalence``. The base
    :meth:`~FrameDict.to_frame` (every column an equal-length array) is exactly
    right here."""


class BootstrapStability(FrameDict):
    """:func:`~topica.evaluate.bootstrap_stability` result. ``topic`` and ``stability``
    are per-topic arrays; ``mean`` (overall) and ``reference`` (the fitted model)
    are scalars, so :meth:`to_frame` frames only the per-topic columns."""

    def to_frame(self):
        """Per-topic stability as a DataFrame (``topic``, ``stability``); the
        scalar ``mean`` and the ``reference`` model stay on the dict."""
        import pandas as pd

        return pd.DataFrame({"topic": self["topic"], "stability": self["stability"]})


class KeywordDiagnostics(FrameDict):
    """:func:`~topica.visualize_keywords` result: ``{set_name: [row, ...]}``.
    :meth:`to_frame` stacks every set's rows into one long frame with a leading
    ``set`` column."""

    _COLUMNS = ["set", "keyword", "count", "proportion", "doc_freq", "in_vocab"]

    def to_frame(self):
        """The per-keyword rows across all sets as one long DataFrame, with a
        ``set`` column naming the keyword set each row came from."""
        import pandas as pd

        rows = [{"set": name, **row} for name, items in self.items() for row in items]
        return pd.DataFrame(rows, columns=self._COLUMNS)


class TimePrevalenceCI(FrameDict):
    """:func:`~topica.time_prevalence_ci` result: ``labels`` plus ``(T, K)``
    arrays ``mean``/``ci_low``/``ci_high``/``sd``. :meth:`to_frame` melts them to
    one row per (period, topic)."""

    _COLUMNS = ["period", "topic", "mean", "ci_low", "ci_high", "sd"]

    def to_frame(self):
        """Long tidy frame, one row per (period, topic), with columns
        ``period``, ``topic``, ``mean``, ``ci_low``, ``ci_high``, ``sd``."""
        import numpy as np
        import pandas as pd

        labels = list(self["labels"])
        mean = np.asarray(self["mean"])
        lo = np.asarray(self["ci_low"])
        hi = np.asarray(self["ci_high"])
        sd = np.asarray(self["sd"])
        T, K = mean.shape
        rows = [
            {
                "period": labels[t],
                "topic": k,
                "mean": float(mean[t, k]),
                "ci_low": float(lo[t, k]),
                "ci_high": float(hi[t, k]),
                "sd": float(sd[t, k]),
            }
            for t in range(T)
            for k in range(K)
        ]
        return pd.DataFrame(rows, columns=self._COLUMNS)
