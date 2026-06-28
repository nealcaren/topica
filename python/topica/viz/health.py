"""The dead / duplicate-topic panel -- candid reporting, and essential for HDP.

Two failure modes a topic model hides from the headline table: **dead topics**
(near-zero expected mass, so they describe almost no documents) and **duplicate
topics** (two topics whose word distributions are near-identical, so they split one
theme in two). Both are routine -- HDP in particular returns many near-zero-mass
topics by construction -- and both are things a methods reviewer expects an author
to have checked. This panel flags them off the same surfaces the rest of the
toolkit uses: ``topic_sizes`` for mass and the topic-word matrix for the nearest
neighbor (phi-cosine, the same metric ``align_topics`` uses).
"""

from __future__ import annotations

import numpy as np

from .base import Panel
from .capability import capabilities
from .terms import _topic_word

_OK = "#4C72B0"
_DEAD = "#BBBBBB"
_DUP = "#C44E52"


class TopicHealth(Panel):
    """Per-topic mass share with dead and duplicate topics flagged.

    A topic is **dead** when its expected mass share falls below ``min_mass_frac``
    (default 1%). A topic is a **duplicate** when its nearest neighbor in phi-cosine
    is at least ``dup_threshold`` similar (default 0.9); the two are reported as a
    pair. The bars are mass share, sorted, colored by flag.
    """

    title = "Topic health"

    def __init__(self, model, *, min_mass_frac=0.01, dup_threshold=0.9):
        from ..analysis import topic_labels, topic_sizes

        self.cap = capabilities(model)
        self.min_mass_frac = float(min_mass_frac)
        self.dup_threshold = float(dup_threshold)

        sizes = topic_sizes(model)
        self._mass = np.asarray(sizes["mass"], dtype=np.float64)
        self._size = np.asarray(sizes["size"], dtype=np.int64)
        self.outliers = int(sizes["outliers"])
        total = max(self._mass.sum(), 1e-12)
        self._share = self._mass / total
        self._labels = topic_labels(model)

        # Nearest neighbor in phi-cosine (the same metric align_topics uses). A
        # topic's nearest neighbor, not the full K x K matrix: cheap and that is
        # all the duplicate flag needs.
        phi = _topic_word(model).astype(np.float64)
        norm = phi / np.clip(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12, None)
        sim = norm @ norm.T
        np.fill_diagonal(sim, -np.inf)
        self._nn = sim.argmax(axis=1)
        self._nn_sim = sim[np.arange(sim.shape[0]), self._nn]

    def _flag(self, t):
        if self._share[t] < self.min_mass_frac:
            return "dead"
        if self._nn_sim[t] >= self.dup_threshold:
            return "duplicate"
        return "ok"

    def to_frame(self):
        import pandas as pd

        k = len(self._mass)
        rows = []
        for t in range(k):
            rows.append({
                "topic": t,
                "label": self._labels[t] if t < len(self._labels) else f"topic_{t}",
                "mass": float(self._mass[t]),
                "mass_frac": float(self._share[t]),
                "hard_count": int(self._size[t]),
                "nearest_topic": int(self._nn[t]),
                "nearest_cosine": float(self._nn_sim[t]),
                "flag": self._flag(t),
            })
        return pd.DataFrame(rows)

    def _figsize(self):
        return (6.5, max(2.5, 0.34 * len(self._mass) + 1.2))

    def _draw(self, fig):
        df = self.to_frame().sort_values("mass_frac")
        k = len(df)
        ax = fig.subplots()
        y = np.arange(k)
        colors = {"ok": _OK, "dead": _DEAD, "duplicate": _DUP}
        ax.barh(y, df["mass_frac"], color=[colors[f] for f in df["flag"]], height=0.7)
        ax.axvline(self.min_mass_frac, color="0.4", lw=1.0, ls="--", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{int(r["topic"])}: {r["label"]}' for _, r in df.iterrows()],
                           fontsize=8)
        ax.set_xlabel("expected mass share")
        n_dead = int((df["flag"] == "dead").sum())
        n_dup = int((df["flag"] == "duplicate").sum())
        parts = [f"{n_dead} dead (<{self.min_mass_frac:.0%})",
                 f"{n_dup} near-duplicate (cos≥{self.dup_threshold:g})"]
        if self.outliers:
            parts.append(f"{self.outliers} outlier docs")
        ax.set_title(f"{self.title} — " + ", ".join(parts), fontsize=9)

    def duplicates(self):
        """The near-duplicate topic pairs as ``[(a, b, cosine), ...]`` (each pair
        once, ``a < b``), sorted by descending similarity."""
        seen = set()
        pairs = []
        for t in range(len(self._mass)):
            if self._nn_sim[t] >= self.dup_threshold:
                a, b = sorted((t, int(self._nn[t])))
                if (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append((a, b, float(self._nn_sim[t])))
        return sorted(pairs, key=lambda p: -p[2])
