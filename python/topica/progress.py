"""Live one-line fit progress: a bar, an ETA, and a sparkline of a fit metric
over time.

``topica.progress()`` returns a *callback* that a model's ``fit`` drives once per
recorded sweep/iteration. It renders a single line that updates in place, so a
long single-threaded fit does not look hung and you can watch the metric converge:

    model.fit(corpus, progress=topica.progress())

    fit  |████████░░░░░░░░| 52%  26/50  ETA 3.1s  ll ▁▂▄▆▇███ -7.81  clusters=30

The callback is model-agnostic. A model calls it as
``callback(iteration, total, info)`` where ``info`` is a dict of named scalar
metrics (e.g. ``{"ll": -7.8, "clusters": 30}``), a single number, or ``None``.
``metric=`` picks which entry to sparkline (default: the first key); the rest are
shown as ``key=value`` postfix. Rendering is dependency-free (Unicode block
sparkline, carriage-return redraw to stderr), so it needs neither tqdm nor a
notebook. You can also drive it yourself from any loop.
"""
from __future__ import annotations

import sys
import time

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=None):
    """A Unicode block sparkline for `values`, normalized over the shown window.

    With `width`, only the last `width` values are drawn (a rolling window, so a
    long fit keeps a fixed-width line). A flat series renders as a low baseline.
    """
    vals = [float(v) for v in values if v is not None and _finite(v)]
    if not vals:
        return ""
    if width is not None and len(vals) > width:
        vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in vals)


def _finite(x):
    return x == x and x not in (float("inf"), float("-inf"))


def _fmt_secs(s):
    if s < 0 or s != s:
        return "?"
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class _ProgressReporter:
    """The stateful callback returned by :func:`progress`. Accumulates the chosen
    metric to draw its sparkline and estimates the remaining time from the mean
    time per completed step."""

    def __init__(self, metric=None, *, width=20, spark_width=40, label="fit", stream=None):
        self.metric = metric
        self.width = width
        self.spark_width = spark_width
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self._history = []
        self._start = None
        self._done = False

    def __call__(self, iteration, total, info=None):
        if self._start is None:
            self._start = time.perf_counter()
        primary, extras = self._split(info)
        if primary is not None:
            self._history.append(primary)
        self._render(iteration, total, primary, extras)
        # Close the line once the last iteration reports.
        if total and iteration >= total and not self._done:
            self._done = True
            self.stream.write("\n")
            self.stream.flush()
        return None

    def _split(self, info):
        """(value_to_sparkline, {label: value} to show as postfix)."""
        if info is None:
            return None, {}
        if isinstance(info, dict):
            if not info:
                return None, {}
            key = self.metric if self.metric in info else next(iter(info))
            primary = info.get(key)
            extras = {k: v for k, v in info.items() if k != key}
            return primary, {key: primary, **extras}
        # a bare scalar
        return info, {}

    def _bar(self, frac):
        frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
        full = int(frac * self.width)
        return "█" * full + "░" * (self.width - full)

    def _render(self, iteration, total, primary, extras):
        elapsed = time.perf_counter() - self._start
        frac = (iteration / total) if total else 0.0
        eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
        parts = [f"{self.label} |{self._bar(frac)}| {int(frac * 100):3d}%"]
        if total:
            parts.append(f"{iteration}/{total}")
        parts.append(f"ETA {_fmt_secs(eta)}")
        if self._history:
            sk = sparkline(self._history, width=self.spark_width)
            key = next(iter(extras)) if extras else "metric"
            parts.append(f"{key} {sk} {_fmt(primary)}")
            rest = list(extras.items())[1:]
        else:
            rest = list(extras.items())
        for k, v in rest:
            parts.append(f"{k}={_fmt(v)}")
        line = "  ".join(parts)
        self.stream.write("\r" + line)
        self.stream.flush()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1e4 else f"{v:.3g}"
    return str(v)


def progress(metric=None, *, width=20, spark_width=40, label="fit", stream=None):
    """Return a live-progress callback for ``fit(progress=...)``.

    `metric` selects which named entry of the model's ``info`` dict to draw as the
    sparkline (default: the first key); the others render as ``key=value``. `width`
    is the bar width; `spark_width` caps the sparkline to a rolling last-N window;
    `label` prefixes the line; `stream` defaults to ``sys.stderr``.

    The returned object is a fresh reporter (keeps its own metric history and
    start time), so use a new ``progress()`` per fit.
    """
    return _ProgressReporter(
        metric, width=width, spark_width=spark_width, label=label, stream=stream
    )
